from flask import Flask,render_template,redirect,url_for,request
import pickle
import json
import numpy as np

app = Flask(__name__)

with open('Bengalore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as fj:
    all_columns = json.load(fj)['data_columns']
columns = all_columns[3:]

def predict_price(sqft, bath, bhk, location):

    x = np.zeros(len(all_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if location in all_columns:
        loc_index = all_columns.index(location)
        x[loc_index] = 1
    pred = round(model.predict([x])[0],2)

    return pred


@app.route('/')
def index():
    return render_template('index.html', columns=columns)


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        sqft = float(request.form['sqft'])
        bath = int(request.form['bath'])
        bhk = int(request.form['bhk'])
        location = str(request.form['location'])

    res = predict_price(sqft, bath, bhk, location)


    return redirect(url_for('result', score=res))


@app.route('/result/<float:score>')
def result(score):
    return render_template('result.html', result=score)


if __name__ == '__main__':
    app.run(debug=True)