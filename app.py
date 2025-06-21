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
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    try:
        data=[float(x) for x in request.form.values()]
        final_input=scalar.transform(np.array(data).reshape(1,-1))
        print(final_input)
        output = regmodel.predict(final_input)
        return render_template("home.html",predicted_text="The House Price Prediction is {}".format(output))
    except ValueError:
        return render_template("home.html", prediction_text="Error: Please enter only numeric values.")

if __name__=="__main__":
    app.run(debug=True)