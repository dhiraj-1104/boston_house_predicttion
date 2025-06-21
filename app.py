# importing libraries
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas

app=Flask(__name__)

# Loading the model
regmodel=pickle.load(open("model/regmodel.pkl",'rb'))
# load the scaling model 
scalar=pickle.load(open('scaling.pkl','rb'))


# Function or api route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Function or api for the predict.
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)