from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import traceback

# initializing flast object
app = Flask(__name__)

# Define a function for Predicting the output for Diabetes and heart disease


def predict(values):
    # For diabetes disease we need 8 feature to predict the output
    if len(values) == 8:
        model = pickle.load(open('Models/diabetes.pkl', 'rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    # For hear disease detection we need 13 feature to predict the output
    elif len(values) == 13:
        model = pickle.load(open('Models/heart.pkl', 'rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

# Home page


@app.route("/")
def home():
    return render_template('home.html')

# Diabetes detection page


@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

# Heart disease detection page


@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

# Malaria detection page


@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

# Cancer detection page


@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('cancer.html')

# Predict page for Printing predicted output of Diabetes and heart disease detection


@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            print(request.form.to_dict())
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            print(to_predict_list)
            pred = predict(to_predict_list)
    except Exception:
        print(traceback.print_exc())
        message = "Please enter valid Data"
        return render_template("home.html", message=message)

    return render_template('predict.html', pred=pred)

# Malaria prediction page


@app.route("/malariapredict", methods=['POST', 'GET'])
def malariapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36, 36))
                img = np.asarray(img)
                img = img.reshape((1, 36, 36, 3))
                img = img.astype(np.float64)
                model = load_model("models/malaria.h5")
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('malaria.html', message=message)
    return render_template('malaria_predict.html', pred=pred)

# Skin Cancer prediction page


@app.route("/cancerpredict", methods=['POST', 'GET'])
def cancerpredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('RGB')
                img = img.resize((224, 224))
                img = np.asarray(img)
                img = img.reshape((-1, 224, 224, 3))
                img = img / 255.0
                model = load_model("models/cancer.h5")
                pred = np.argmax(model.predict(img)[0])
        except Exception:
            traceback.print_exc()
            message = "Please upload an Image"
            return render_template('cancer.html', message=message)
    return render_template('cancer_predict.html', pred=pred)


# Main function
if __name__ == '__main__':
    app.run(debug=True)
