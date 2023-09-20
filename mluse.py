import pickle

import numpy as np
from flask import Flask, render_template, request

# Create a flask app
flask_app = Flask(__name__)
with open('model.pkl', 'rb') as file1:
    model= pickle.load(file1)

@flask_app.route("/")
def Home():
    return render_template("home.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("home.html", prediction_text = "The flower species is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run('0.0.0.0' ,debug=True)