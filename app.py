import numpy as np
from flask import Flask, request, jsonify
import pickle
import math

app = Flask(__name__)
with open('breast_cancer.pkl', 'rb') as f:
    model, label_encoder = pickle.load(f)

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
    return jsonify({'prediction_text': "Predicted stage: {}".format(predicted_stage)})
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
