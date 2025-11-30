import numpy as np
from tensorflow import keras
from image_recognizer import preprocess_example
from PIL import Image
import io

# load model & labels
MODEL_PATH = "models/road_issues_best.keras"
LABEL_PATH = "results/class_names.npy"

model = keras.models.load_model(MODEL_PATH)
class_names = np.load(LABEL_PATH, allow_pickle=True).tolist()

IMG_SIZE = (224, 224)

def _preprocess_image_bytes(image_bytes, target_size=(224, 224)):
    """
    Robust preprocessing for model input:
    - accepts raw bytes or file-like
    - ensures RGB, fixed size, float32, normalized [0,1]
    - returns shape (1, H, W, 3)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        # fallback if caller passed a path / fileobj
        img = Image.open(image_bytes)

    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize(target_size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    arr /= 255.0  # normalize to [0,1]
    arr = np.expand_dims(arr, axis=0)
    return arr

# return prediction of image
def classify_image(img):
    x = _preprocess_image_bytes(img, target_size=(224, 224))
    # ensure dtype float32
    x = x.astype(np.float32, copy=False) # model includes preprocess layers
    
    preds = model.predict(x)[0]
    class_index = int(np.argmax(preds))
    confidence = float(preds[class_index])
    category = class_names[class_index]

    return {
        "category": category,
        "confidence": confidence,
        "index": class_index,
    }
