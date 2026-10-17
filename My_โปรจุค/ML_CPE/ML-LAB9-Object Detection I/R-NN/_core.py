from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL_PATH = OUT / "model.keras"

CLASSES = [
    "person", "animal", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
]
IMG_SIZE = 64

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first using train.py")
    return tf.keras.models.load_model(MODEL_PATH)

def detect_demo(model, frame, threshold=0.85, regions=25):
    h, w = frame.shape[:2]
    predictions = []
    
    crop = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(crop.astype(np.float32) / 255.0, axis=0)
    
    preds = model.predict(x, verbose=0)[0]
    cls = int(np.argmax(preds))
    conf = float(np.max(preds))
    
    if conf >= threshold:
        predictions.append((int(h*0.2), int(w*0.2), int(h*0.8), int(w*0.8), cls, conf))
        
    return predictions

def draw(frame, preds):
    for box in preds:
        y1, x1, y2, x2, cls, conf = box
        label = f"{CLASSES[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame