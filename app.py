from flask import Flask, jsonify, request, render_template
import pandas as pd
import pickle

app = Flask("__name__")

@app.route('/', methods = ["GET"])
def home():
    return render_template("index.html")


def get_cleaned_data(form_data):

    gestation = int(form_data["gestation"])
    parity = float(form_data["parity"])
    age = float(form_data["age"])
    height = float(form_data["height"])
    weight = float(form_data["weight"])
    smoke = float(form_data["smoke"])

    cleaned_data = {"gestation":[gestation],
                    "parity":[parity],
                    "age":[age],
                    "height":[height],
                    "weight":[weight],
                    "smoke":[smoke]
                    }
    return cleaned_data

@app.route('/predict', methods = ["POST"])
def get_prediction():
    baby_data_form = request.form
    baby_data_cleaned = get_cleaned_data(baby_data_form)
    b_df = pd.DataFrame(baby_data_cleaned)



    with open("model.pkl", 'rb') as obj:
        mymodel = pickle.load(obj)

        prediction = mymodel.predict(b_df)
        prediction = round(float(prediction[0]), 2)

        response = {"PREDICTION":prediction}
        return render_template("index.html", response = response)
    
if __name__ == "__main__":
    app.run(debug = True)