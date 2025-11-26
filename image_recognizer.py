import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from datasets import load_from_disk, load_dataset
from keras import layers, models, Input, optimizers
from keras.applications import ResNet50
from datasets import DatasetDict
import numpy as np
from PIL import Image
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
import json 

#KEY Variables
IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42
MAX_TRIALS = 5

#Coverts the grayscale image to RGB since dataset is mixed
def preprocess_example(example):
    img = example["image"]
    # make sure it's a PIL image
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))

    img = img.convert("RGB")                  # handle grayscale
    img = img.resize(IMG_SIZE)               # unify size
    example["image"] = np.array(img)
    return example

#Hyparameter tuner
def build_model(hp: kt.HyperParameters, num_classes):
    #Test different sets of CNN architectures
    backbone_name = hp.Choice(
        "backbone",
        ["resnet50", "densenet121"],
    )
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomContrast(0.1),
        ],
        name="augment",
    )

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)

    if backbone_name == "resnet50":
        x = keras.applications.resnet.preprocess_input(x)
        base = keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_tensor=x
        )
    elif backbone_name == "densenet121":
        x = keras.applications.densenet.preprocess_input(x)
        base = keras.applications.DenseNet121(
            include_top=False, weights="imagenet", input_tensor=x
        )


    fine_tune = hp.Boolean("fine_tune")
    if not fine_tune:
        base.trainable = False
    else:
        for layer in base.layers[:-20]:
            layer.trainable = False
        for layer in base.layers[-20:]:
            layer.trainable = True

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)

    num_dense_layers = hp.Int("num_dense_layers", 1, 3)
    for i in range(num_dense_layers):
        units = hp.Int(f"dense_units_{i}", 64, 512, step=64)
        act = hp.Choice(f"activation_{i}", ["relu", "gelu", "selu"])
        x = layers.Dense(units, activation=act)(x)

        drop = hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.1)
        if drop > 0:
            x = layers.Dropout(drop)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    lr = hp.Float("lr", 1e-5, 3e-4, sampling="log")
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    #Use load from disk if you already have the dataset from hugging face loaded else 
    # use the load dataset 
    # ds = load_dataset("Programmer-RD-AI/road-issues-detection-dataset")
    # ds.save_to_disk('service_dataset')
    ds = load_from_disk('service_dataset//train')

    class_names = ds.features["label"].names
    base = ds

    # First split: 80% train+val, 20% test
    trainval_test = base.train_test_split(test_size=0.2, seed=SEED)
    train = trainval_test["train"]
    test = trainval_test["test"]

    # Second split inside train+val: 80% train, 20% val
    train_val = train.train_test_split(test_size=0.2, seed=SEED)
    sub_train = train_val["train"]
    valid_set   = train_val["test"]

    splits = DatasetDict({
        "train": sub_train,
        "val": valid_set,
        "test": test,
    })


    # apply to all splits
    splits = splits.map(preprocess_example, num_proc=4)

    # tell HF we want TensorFlow format
    for split_name in ["train", "val", "test"]:
        splits[split_name] = splits[split_name].with_format(
            type="tensorflow",
            columns=["image", "label"],
        )
    train_ds = splits["train"].to_tf_dataset(
        columns=["image"],
        label_cols=["label"],
        shuffle=True,
        batch_size=BATCH,
    )

    val_ds = splits["val"].to_tf_dataset(
        columns=["image"],
        label_cols=["label"],
        shuffle=False,
        batch_size=BATCH,
    )

    test_ds = splits["test"].to_tf_dataset(
        columns=["image"],
        label_cols=["label"],
        shuffle=False,
        batch_size=BATCH,
    )

    num_classes = splits["train"].features["label"].num_classes
    
    tuner = kt.BayesianOptimization(
        lambda hp: build_model(hp, num_classes),
        objective="val_accuracy",
        max_trials=MAX_TRIALS,
        directory="image_identifier",
        project_name="city_issues",
        overwrite=True,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
        )
    ]

    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=12,
        callbacks=callbacks,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.hypermodel.build(best_hp)

    # Optionally train longer on train+val then evaluate on test_ds
    train_val_ds = train_ds.concatenate(val_ds)
    best_model.fit(train_val_ds, epochs=20, callbacks=callbacks)
    test_loss, test_acc = best_model.evaluate(test_ds)
    print("Test accuracy:", test_acc)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    
    # Save best trained model
    best_model.save("models/road_issues_best.keras")
    print("Saved model → models/road_issues_best.keras")

    
    # Save best hyperparameters
    with open("results/best_hparams.json", "w") as f:
        json.dump(best_hp.values, f, indent=2)
    print("Saved hyperparameters → results/best_hparams.json")

    
    # Save class names
    np.save("results/class_names.npy", np.array(class_names))
    print("Saved class names → results/class_names.npy")

    
    # Save final test accuracy
    run_summary = {
        "test_accuracy": float(test_acc),
        "num_classes": len(class_names),
    }
    with open("results/run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)
    print("Saved test accuracy → results/run_summary.json")


if __name__ == "__main__":
    main()