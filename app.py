import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the pre-trained model and label encoder
with open('breast_cancer.pkl', 'rb') as f:
    model, label_encoder = pickle.load(f)
@app.route('/', methods=['GET','POST'])
def home():
    return jsonify({'message': 'Send a POST request to /predict endpoint with required data for prediction.'})
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.json
    
    # Extract features from the JSON data
    input_data = [int(data['Race']), int(data['Marital Status']), int(data['N Stage']), int(data['6th Stage']), 
                  int(data['Differentiate']), int(data['Grade']), int(data['A Stage']), 
                  int(data['Estrogen Status']), int(data['Progesterone Status']), int(data['Age']), 
                  int(data['Tumor_Size']), int(data['Regional Node Examined']), int(data['Reginol Node Positive']), 
                  int(data['Survival Months']), int(data['Breast Cancer History'])]
    
    # Predict the stage using the model
    prediction = model.predict([input_data])
    
    # Convert the predicted stage back to its original label
    predicted_stage = label_encoder.inverse_transform(prediction)
    
    # Return the prediction as JSON response
    return jsonify({'predicted_stage': predicted_stage[0]})
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
