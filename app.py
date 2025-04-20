# app.py
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import base64
import io
from PIL import Image
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()


# Load model paths
model_paths = {
    "model_a": os.getenv("MODEL_A_PATH"),
    "model_b": os.getenv("MODEL_B_PATH"),
}

# Load face detector
face_classifier = cv2.CascadeClassifier(os.getenv("CASCADE_PATH"))
class_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# Cache models to avoid reloading
loaded_models = {}

def get_model(model_key):
    if model_key in loaded_models:
        return loaded_models[model_key]
    if model_key in model_paths:
        model = load_model(model_paths[model_key])
        loaded_models[model_key] = model
        return model
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = data['image']
    model_key = data.get('model')  # e.g., 'model_a' or 'model_b'

    classifier = get_model(model_key)
    if classifier is None:
        return jsonify({'error': 'Invalid model selected.'}), 400

    encoded_data = img_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            return jsonify({'emotion': label})

    return jsonify({'emotion': 'No Face Found'})

if __name__ == '__main__':
    app.run(debug=True)
