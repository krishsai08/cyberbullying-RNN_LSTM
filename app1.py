import os
import re
import contractions
import spacy
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import pickle

# Initialize Flask app
app = Flask(__name__)

# Define relative model and tokenizer paths
MODEL_PATH_RNN = "models/rnn_model.h5"
MODEL_PATH_LSTM = "models/lstm_model.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

# Load the RNN and LSTM models and tokenizer
try:
    rnn_model = load_model(MODEL_PATH_RNN)
    lstm_model = load_model(MODEL_PATH_LSTM)
    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)
except Exception as e:
    raise RuntimeError(f"Error loading model or tokenizer: {e}")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise RuntimeError(f"Error loading spaCy model: {e}")

# Stopwords for preprocessing
try:
    stop_words = set(stopwords.words("english"))
except Exception as e:
    import nltk
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# Preprocessing function
def preprocess_comment(comment):
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Expand contractions
        comment = contractions.fix(comment)

        # Remove punctuation and digits
        comment = re.sub(r"[^\w\s]", "", comment)  # Remove punctuation
        comment = "".join([c for c in comment if not c.isdigit()])  # Remove digits

        # Remove stopwords
        comment = " ".join([word for word in comment.split() if word not in stop_words])

        # Lemmatize with spaCy
        doc = nlp(comment)
        comment = " ".join(token.lemma_ for token in doc)

        return comment

    except Exception as e:
        raise ValueError(f"Error during preprocessing: {e}")

# Define routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.json or "comment" not in request.json:
            return jsonify({"error": "Invalid input. Please send a JSON with a 'comment' field."}), 400

        user_comment = request.json["comment"]

        # Preprocess the comment
        preprocessed_comment = preprocess_comment(user_comment)

        # Tokenize and pad the comment
        comment_sequence = tokenizer.texts_to_sequences([preprocessed_comment])
        padded_comment = pad_sequences(comment_sequence, maxlen=100, padding='pre')

        # Predict using both RNN and LSTM models
        rnn_prediction = rnn_model.predict(padded_comment)
        lstm_prediction = lstm_model.predict(padded_comment)

        # Average the probabilities from both models
        avg_prediction = (rnn_prediction[0][0] + lstm_prediction[0][0]) / 2

        # Convert the averaged probability to a class (0 or 1) based on threshold
        final_prediction = 1 if avg_prediction >= 0.5 else 0

        # Determine the final prediction label
        prediction_label = "Cyberbullying" if final_prediction == 1 else "Not Cyberbullying"

        # Prepare and return the result
        result = {"Classification": prediction_label}
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {e}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
