from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)
model = tf.keras.models.load_model("model/Final.h5")
labels = ["Cat", "Cow", "Dog", "Fish", "Goat"]
animal_info = [
    ["Cat", "15 years", "Carnivore (meat, fish)", "Homes, urban areas"],
    ["Cow", "18–22 years", "Herbivore (grass, hay)", "Farms, pastures"],
    ["Dog", "10–13 years", "Omnivore (meat, rice, kibble)", "Homes, outdoors"],
    ["Fish", "1–5 years", "Omnivore (flakes, pellets)", "Aquariums, freshwater"],
    ["Goat", "15–18 years", "Herbivore (grass, shrubs)", "Hills, farms"]
]

@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        img = request.files.get("camera_img") or request.files.get("file_img")

        if img and img.filename != "":
            image = Image.open(img).convert("RGB")
            image = image.resize((224, 224))
            img_array = np.array(image)
            img_array = preprocess_input(img_array)  
            img_array = np.expand_dims(img_array, axis=0)  

            # Predicting
            predictions = model.predict(img_array)
            pred_index = np.argmax(predictions[0])
            prediction_label = labels[pred_index]

            print("Raw prediction output:", predictions)
            print("Predicted index:", pred_index)
            print("Predicted label:", prediction_label)

            return render_template("result.html", prediction=prediction_label,info=animal_info[pred_index])
        else:
            return render_template("index.html", prediction="No image uploaded.")

    return render_template("index.html")

if __name__ == "__main__":
    app.run( debug=True)

