from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model("brain_tumor_model.h5")

def preprocess(image):
    image = image.resize((224,224))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file)

    img = preprocess(image)
    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        result = "Tumor Detected ❌"
    else:
        result = "No Tumor ✅"

    return jsonify({"result": result})

app.run(debug=True)
