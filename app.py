from crypt import methods
import pickle
from pyexpat import model
from flask import Flask, request, app, jsonify, url_for, render_template

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix

app = Flask(__name__)
p_model= pickle.load(open('knn_model.pkl','rb'))

@app.route('/')

def home:
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])

def predict api():
    data = request.json['data']
    print(data)


