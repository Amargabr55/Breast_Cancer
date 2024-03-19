import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)
with open('breast_cancer.pkl', 'rb') as f:
    model, label_encoder = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = [int(data['Race']), int(data['Marital Status']), int(data['N Stage']), int(data['6th Stage']), 
                  int(data['differentiate']), int(data['Grade']), int(data['A Stage']), 
                  int(data['Estrogen Status']), int(data['Progesterone Status']), int(data['Age']), 
                  int(data['Tumor_Size']), int(data['Regional Node Examined']), int(data['Reginol Node Positive']), 
                  int(data['Survival Months']), int(data['breast_cancer_history'])]
    prediction = model.predict([input_data])
    predicted_stage = label_encoder.inverse_transform(prediction)
    return render_template('index.html', prediction_text="Predicted stage: {}".format(predicted_stage))
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
"""
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
with open('breast_cancer.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
import requests   
import json 

app = Flask(__name__)

url = 'http://127.0.0.1:5000/predict'
# Define endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Assuming the incoming data is a list of features
    features = data['features']
    # Convert features to numpy array and make prediction
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    # Convert prediction to a JSON serializable format (Python list)
    prediction_list = prediction.tolist()
    return jsonify({'prediction': prediction_list})
    json_data = json.dumps(data)
    response = requests.post(url, json=json_data)
    
    if response.status_code == 200:
        prediction = response.json()['prediction']
        print("Prediction:", prediction)
    else:
        print("Error:", response.status_code)
        



if __name__ == '__main__':
    app.run()
    """