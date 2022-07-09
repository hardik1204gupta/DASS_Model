import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app=Flask(__name__)
Depression_Model=pickle.load(open('depression.pkl','rb'))
Anxiety_Model=pickle.load(open('anxiety.pkl','rb'))
Stress_Model=pickle.load(open('stress.pkl','rb'))
l=[]

@app.route('/')
def first():
    return render_template('home.html')

@app.route('/form')
def home():
    return render_template('form.html')

@app.route('/predict',methods=['POST'])
def dass_predictor():
    int_features=[int(x) for x in request.form.values()]
    for i in int_features:
        l.append(i)
        
    pickle.dump(l,open('test.pkl','wb'))
    
    # depression question Q3, 5, 10, 13, 16, 17, 21
    depression_features=np.array([int_features[2],int_features[4],int_features[9],int_features[12],int_features[15],int_features[16],int_features[20]])
    
    # anxiety question Q2, 4, 7, 9, 15, 19, 20
    anxiety_features=np.array([int_features[1],int_features[3],int_features[6],
                               int_features[8],int_features[14],int_features[18],
                               int_features[19]])

    # stress question Q1, 6, 8, 11, 12, 14, 18
    stress_features=np.array([int_features[0],int_features[5],int_features[7],
                              int_features[10],int_features[11],int_features[13],
                              int_features[17]])

    # depression prediction
    depression_prediction=Depression_Model.predict(depression_features.reshape(1,-1))
    depression_output=depression_prediction[0]

    # anxiety prediction
    anxiety_prediction=Anxiety_Model.predict(anxiety_features.reshape(1,-1))
    anxiety_output=anxiety_prediction[0]

    # stress prediction
    stress_prediction=Stress_Model.predict(stress_features.reshape(1,-1))
    stress_output=stress_prediction[0]

    return render_template('index.html',prediction_text1='{}'.format(depression_output),
                           prediction_text2='{}'.format(anxiety_output),
                           prediction_text3='{}'.format(stress_output))

if __name__=="__main__":
    app.run(debug=True)
