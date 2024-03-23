import json
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the pre-trained model and label encoder
with open('breast_cancer.pkl', 'rb') as f:
    model, label_encoder = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Send a POST request to /predict endpoint with required data for prediction.'})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Simulated JSON data for prediction
        data = {
        'Race': int(input("Enter Race: ")),
        'Marital Status': int(input("Enter Marital Status: ")),
        'N Stage': int(input("Enter N Stage: ")),
        '6th Stage': int(input("Enter 6th Stage: ")),
        'Differentiate': int(input("Enter Differentiate: ")),
        'Grade': int(input("Enter Grade: ")),
        'A Stage': int(input("Enter A Stage: ")),
        'Estrogen Status': int(input("Enter Estrogen Status: ")),
        'Progesterone Status': int(input("Enter Progesterone Status: ")),
        'Age': int(input("Enter Age: ")),
        'Tumor_Size': int(input("Enter Tumor Size: ")),
        'Regional Node Examined': int(input("Enter Regional Node Examined: ")),
        'Reginol Node Positive': int(input("Enter Reginol Node Positive: ")),
        'Survival Months': int(input("Enter Survival Months: ")),
        'Breast Cancer History': int(input("Enter Breast Cancer History: "))
    }

        # Check if all required fields are present
        required_fields = [
            'Race', 'Marital Status', 'N Stage', '6th Stage',
            'Differentiate', 'Grade', 'A Stage',
            'Estrogen Status', 'Progesterone Status', 'Age',
            'Tumor_Size', 'Regional Node Examined', 'Reginol Node Positive',
            'Survival Months', 'Breast Cancer History'
        ]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400
        
        # Extract features from the JSON data
        input_data = [data[field] for field in required_fields]
        
        # Predict the stage using the model
        prediction = model.predict([input_data])
        
        # Convert the predicted stage back to its original label
        predicted_stage = label_encoder.inverse_transform(prediction)
        
        # Return the prediction as JSON response
        response_data = {"predicted_stage": predicted_stage[0]}
        
        return jsonify(response_data)
    else:
        return jsonify({'error': 'Method Not Allowed'}), 405

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
