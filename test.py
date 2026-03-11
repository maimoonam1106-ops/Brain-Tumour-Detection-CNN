import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("brain_tumor_model.h5")

def preprocess(image):
    image = image.resize((224,224))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)
    return image

# Give image path here
img_path = "testing/yes/pituitary/Te-pi_2.jpg"

image = Image.open(img_path)

img = preprocess(image)

prediction = model.predict(img)[0][0]

if prediction > 0.5:
    print("Tumor Detected ❌")
else:
    print("No Tumor ✅")
