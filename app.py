''' attention ici on appelle request et non pas requests (sans "s")'''
from flask import Flask, render_template, url_for, request
from sklearn.externals import joblib
import numpy as np

# =============================================================================
# salary
# =============================================================================

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict" , methods = ["POST", "GET"])
def predict():
    
    regressor = joblib.load("./linear_regression_model.pkl")
    year = [[float(dict(request.form)["year"])]]
    pred = regressor.predict(year)
    return render_template("predict.html", pred = pred)

if __name__ == "__main__":
    app.run(debug=True)
