from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this line

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import io
from PIL import Image

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load the trained pneumonia detection model
model = load_model("chest_xray_v1.h5")

def preprocess_image(img):
    # Ensure the image is in RGB mode
    img = img.convert("RGB")  

    # Resize the image
    img = img.resize((224, 224))

    # Convert to numpy array
    img = image.img_to_array(img)

    # Expand dimensions to match model input shape
    img = np.expand_dims(img, axis=0)

    # Apply preprocessing specific to the model
    img = preprocess_input(img)

    return img

@app.route("/", methods=["GET"])
def home():
    return "Flask is running! Go to /predict to make a prediction."

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read()))
    
    img = preprocess_image(img)

    prediction = model.predict(img)
    #result = "Person is Affected by PNEUMONIA" if int(prediction[0][0]) == 0 else "Result is Normal"
    result =""
    if prediction[0][1]<0.5:
        result = f"Result is Normal {round((1-prediction[0][1])*100)}%"
    else:
        result = f"Person is Affected By PNEUMONIA  {round(prediction[0][1]*100)}%"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
