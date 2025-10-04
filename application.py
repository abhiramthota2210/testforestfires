from flask import Flask,request,jsonify,render_template
import pickle
import numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

app.config['DEBUG'] = True
## import ridge regression model and standard scaler pickle

ridge_model=pickle.load(open("end_to_end/models/ridge.pkl","rb"))
scaler=pickle.load(open("end_to_end/models/scaler.pkl","rb"))



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict_datapoint():
    if request.method == 'POST':
        pass
        # 1. get the data from post request

        Temperature=request.form.get('Temperature')
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        input_df = pd.DataFrame([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]],
                                columns=['Temperature','RH','Ws','Rain','FFMC','DMC','ISI','Classes','Region'])

        # 3. Scale and predict
        scaled_data = scaler.transform(input_df)
        result = ridge_model.predict(scaled_data)
        return render_template('home.html', results=result[0])
        
    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0")
