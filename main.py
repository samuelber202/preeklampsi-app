from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 17)
    header = ['Age', 'Height', 'Weight', 'Gestational Age', 'Gravida', 'Parity', 'Abortus', 'Systolic',  'Diastolic', 'Proteinuria', 'Delivery Method', 'Creatinine', 'Hemoglobin','Leukocytes', 'Hematocrit', 'Platelets', 'Erythrocytes']
    df = pd.DataFrame(to_predict, columns=header)

    # convert height, weight to bmi
    df['BMI'] =  df['Weight'] / ((df['Height'] / 100)**2)
    df.drop(['Height', 'Weight'], axis=1, inplace=True)
    df.insert(1, 'BMI', df.pop('BMI'))

    loaded_model = joblib.load('naive_bayes_model.joblib')
    result = loaded_model.predict(df)
    return result[0]

@app.route('/')
def index():
    return render_template("index.j2")

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        name = to_predict_list.pop(0)
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)        
        if int(result)== 1:
            prediction ='Severe Preeclampsia'
        else:
            prediction ='Moderate Preeclampsia'           
        return render_template("result.j2", name=name, prediction = prediction)

if __name__ == "__main__":
    app.run(port=5000, debug=True)