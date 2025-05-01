from flask import Flask, request, jsonify
import gzip
import pickle
import os

app = Flask(__name__)

CLASSES = {
    0: "negative",
    4: "positive"
}

def load_model():
    model_filename = "data/model.dat.gz"
    if not os.path.isfile(model_filename):
        raise FileNotFoundError("Model file not found.")
    with gzip.open(model_filename, 'rb') as fmodel:
        return pickle.load(fmodel, encoding='latin1')

model = load_model()

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    text = request.json.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    x_vector = model.vectorizer.transform([text])
    y_predicted = model.predict(x_vector)
    result = CLASSES.get(y_predicted[0], "unknown")
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
