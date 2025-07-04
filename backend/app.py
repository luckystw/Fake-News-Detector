import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
from flask import Flask, request, jsonify
import torch
from flask_cors import CORS
import pickle
from transformers.pipelines import pipeline

app = Flask(__name__)
CORS(app)

# Load classic ML models and vectorizer
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)

model_files = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'Gradient Boost': 'gradient_boost.pkl',
    'Random Forest': 'random_forest.pkl'
}
ml_models = {}
for name, fname in model_files.items():
    with open(os.path.join(MODEL_DIR, fname), 'rb') as f:
        ml_models[name] = pickle.load(f)

# Load transformer pipeline (for demo, zero-shot)
LABELS = ['True', 'False', 'Mostly False', 'Misleading']
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    # Classic ML predictions
    X_vec = vectorizer.transform([text])
    ml_results = {}
    for name, model in ml_models.items():
        pred = model.predict(X_vec)[0]
        ml_results[name] = 'Real' if pred == 1 else 'Fake'
    # Transformer prediction
    result = classifier(text, LABELS)
    transformer_label = result['labels'][0] if isinstance(result, dict) and 'labels' in result else str(result)
    return jsonify({
        'ml_results': ml_results,
        'transformer_result': transformer_label
    })

if __name__ == '__main__':
    app.run(debug=True) 