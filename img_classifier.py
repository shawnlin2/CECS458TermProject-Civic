import numpy as np
from tensorflow import keras
from image_recognizer import preprocess_example

# load model & labels
MODEL_PATH = "models/road_issues_best.keras"
LABEL_PATH = "results/class_names.npy"

model = keras.models.load_model(MODEL_PATH)
class_names = np.load(LABEL_PATH, allow_pickle=True).tolist()

IMG_SIZE = (224, 224)

# return prediction of image
def classify_image(img):
    x = preprocess_example(img)
    preds = model.predict(x)[0]

    class_index = int(np.argmax(preds))
    confidence = float(preds[class_index])
    category = class_names[class_index]

    return {
        "category": category,
        "confidence": confidence,
        "index": class_index,
    }
