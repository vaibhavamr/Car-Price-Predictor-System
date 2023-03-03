from flask import Flask, render_template, request
import numpy as np


app = Flask(__name__)
import joblib
model = joblib.load('Model_Cpp.pkl')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = float(request.form['const'])
    val2 = float(request.form['horsepower'])
    val3 = float(request.form['carwidth'])
    val4 = float(request.form['hatchback'])
    val5 = float(request.form['Highend'])

    arr = np.array([val1, val2, val3, val4, val5])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])

    return render_template('index.html', data=int(pred))
if __name__ == '__main__':
    app.run(port=5002,debug=True)
