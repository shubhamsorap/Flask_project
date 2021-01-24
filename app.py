from flask import Flask, render_template, request
#import predictActivity

import pandas as pd
import numpy as np 
import sys
import logging

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

@app.route('/')
def home():
    return render_template("Input.html")


@app.route('/prediction')
def predict():
    vari = request.args.get('slept')
    data = pd.read_csv('train_data.csv')
    

    X = data[['Hours Slept']]
    y = data['Active']

    from sklearn.model_selection import train_test_split

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

    from sklearn.linear_model import LinearRegression

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    ans = reg.predict(np.array([int(vari)]).reshape(-1,1))[0]
    return request.args.get('name') + " was active for : " + str(np.round(ans,0)) + " hours!!"


if __name__ == "__main__":
    app.debug = True
    app.run()
