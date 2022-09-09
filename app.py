import json
import flask
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix

app = Flask(__name__)
model= pickle.load(open('model.pkl','rb'))
scalar= pickle.load(open('scaling.pkl','rb'))

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])

def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data= scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output= model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    if output == 0:
        return render_template("home.html",prediction_text="REGULAR, with a price range of Rs 5000 and Rs 8000")
    elif output == 1:
        return render_template("home.html",prediction_text="AFFORDABLE, with a price range of Rs 8000 and Rs 12000")
    elif output == 2:
        return render_template("home.html",prediction_text="COSTLY, with a price range of Rs 12000 and Rs 15000")
    elif output == 3:
        return render_template("home.html",prediction_text="VERY COSTLY, with a price range greater than Rs 15000")

    

if __name__=="__main__":
    app.run(debug=True )



