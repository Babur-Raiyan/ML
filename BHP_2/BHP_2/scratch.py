import json
import pickle
import numpy as np

with open('Bengalore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as fj:
    all_columns = json.load(fj)['data_columns']
columns = all_columns[3:]
#print(columns)
#print(all_columns)

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
