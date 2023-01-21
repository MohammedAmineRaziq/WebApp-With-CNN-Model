import numpy as np
from flask import Flask, render_template, request
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model("static/models/best_model.h5")


@app.route("/")
def index():
    return render_template("index.html", display=False)


@app.route("/process_image", methods=["POST"])
def process_image():
    image = request.files["image"]
    extension = image.filename.split(".")[1]
    filename = "IMAGE." + extension
    image.save("static/uploads/images/" + filename)
    return render_template("index.html", class_=predict()[0], probability=predict()[1],
                           image="static/uploads/images/" + filename, display=True)


def predict():
    image = ImageDataGenerator(rescale=1./255).flow_from_directory(
        "static/uploads/",
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary",
        shuffle=False)
    probability = model.predict(image)
    print(probability)
    if probability[0][0] > 0.5:
        return ["Recyclable", round(probability[0][0] * 100)]
    else:
        return ["Organic", 100 - round(probability[0][0] * 100)]


app.run()
