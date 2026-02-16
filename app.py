from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle, numpy as np, re

# This version is optimized for local deployment (e.g., VS Code or Pycharm).
# It relies on your local environment having the model/tokenizer files and a 'templates' folder.

app = Flask(__name__)

try:
    # Use the filename you provided
    model = load_model("SentimentModel_balanced (1).keras")
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    print("Model and Tokenizer loaded successfully.")
except Exception as e:
    # Informative message for local debugging
    print(f"ERROR: Failed to load model or tokenizer. Ensure files are in the correct directory: {e}")
    model, tokenizer = None, None # Set to None for graceful error handling

maxlen = 400
# The label_map is not used as we are implementing custom thresholds for ternary classification.
# label_map = {0:'Negative', 1:'Neutral', 2:'Positive'} 

def clean_text(text):
    """Performs text cleaning steps needed before tokenization."""
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text) # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation and special characters
    return re.sub(r'\s+', ' ', text).strip() # Normalize whitespace

@app.route("/")
def home():
    # Flask looks for 'index.html' in the 'templates' folder automatically.
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or tokenizer is None:
        # Return an error message if the model failed to load at startup
        return jsonify({"sentiment": "Error", "confidence": "0.00", "message": "Model not loaded. Check server console for details."}), 500

    # 1. Get user input
    # The client-side JS sends form data (application/x-www-form-urlencoded)
    msg = request.form["message"]
    
    # 2. Preprocess the text
    seq = tokenizer.texts_to_sequences([clean_text(msg)])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
    
    # 3. Predict the positive probability score (assuming Sigmoid output)
    # verbose=0 suppresses Keras logging output during prediction
    pred_score = model.predict(padded, verbose=0)[0][0]
    
    # 4. Apply custom thresholds for ternary output (Positive, Neutral, Negative)
    # Thresholds are 0.5 for Positive and 0.1 for the Neutral/Negative boundary.
    if pred_score >= 0.5:
        sentiment = 'Positive'
    elif pred_score >= 0.1: 
        sentiment = 'Neutral'
    else: # pred_score < 0.1
        sentiment = 'Negative'
        
    # The confidence reported is the raw positive score (0 to 1), formatted to 2 decimal places.
    conf = float(pred_score)
    
    return jsonify({"sentiment": sentiment, "confidence": f"{conf:.2f}"})

if __name__ == "__main__":
    # Standard Flask command for running locally in debug mode
    # Access this locally at http://127.0.0.1:5000/
    app.run(debug=True)