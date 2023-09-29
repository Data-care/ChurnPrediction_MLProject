from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np


app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

model = loaded_data['model']
encoder1 = loaded_data['label_encoder1']
encoder2 = loaded_data['label_encoder2']
scale = loaded_data['scaler']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        credit_score = int(request.form['credit_score'])
        country = request.form['country']
        gender = request.form['gender']
        age = int(request.form['age'])
        tenure = int(request.form['tenure'])
        balance = float(request.form['balance'])
        products_number = int(request.form['products_number'])
        credit_card = int(request.form['credit_card'])
        active_member = int(request.form['active_member'])
        estimated_salary = float(request.form['estimated_salary'])

        input_data = [[credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member,estimated_salary]]


        columns = ['credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 'products_number', 'credit_card',
                   'active_member', 'estimated_salary']


        dtype_dict = {'credit_score': float, 'age': int, 'tenure': int, 'balance': float, 'products_number': int,'credit_card': int, 'active_member': int, 'estimated_salary': float}

        new_df = pd.DataFrame(input_data, columns=columns)

        new_df = new_df.astype(dtype_dict)

        new_df['country'] = encoder2.transform(new_df['country'])
        new_df['gender'] = encoder1.transform(new_df['gender'])

        scaled = scale.transform(new_df)

        prediction = model.predict(scaled)

        if prediction[0] == 1:
            result = "Chances of Churn: Yes"
        else:
            result = "Chances of Churn: No"

        return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
