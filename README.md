# Cyberbullying Comment Classification

This project is a web-based application designed to classify comments into two categories: **Cyberbullying** and **Not Cyberbullying**. The application uses machine learning models to analyze comments and predict whether they are instances of cyberbullying, specifically focusing on harmful or offensive language.

## Features

- **Preprocessing**: Text is cleaned, lemmatized, and filtered to remove stopwords and unwanted characters.
- **Machine Learning Models**:
  - Logistic Regression
  - Random Forest Classifier
  - Support Vector MAchines
  - RNN
  - LSTM
- **Interactive Web Interface**: Users can input a comment, and the application predicts whether the comment is cyberbullying.
- **Real-time Predictions**: Powered by Flask, predictions are returned instantly upon submission.

## Purpose

The primary purpose of this project is to demonstrate the potential of machine learning models in identifying harmful online behavior, which can assist in moderating content and promoting a safer online environment.

---

## Technology Stack

- **Backend**: Python with Flask
- **Frontend**: HTML, CSS, and JavaScript (with Fetch API for async communication)
- **Models**: Pre-trained models stored in `.joblib` format
- **Libraries**:
  - Flask
  - scikit-learn
  - nltk
  - spacy
  - joblib
  - contractions
  - re

---

## Installation and Setup

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

### Steps to Run Locally

1. Clone the repository:
   

Install dependencies:

pip install -r requirements.txt

Download the spaCy model:

python -m spacy download en_core_web_sm
Place the pre-trained models in the models directory:

vectorizer.joblib
lr_model.joblib
rfc_model.joblib

Run the application:

python app.py
Open your browser and navigate to:
http://127.0.0.1:5000/

Usage
Enter a comment in the input box on the webpage.
Click the "Classify" button.
View the predictions from both models in a styled table:
Logistic Regression: Displays whether the comment is "Cyberbullying" or "Not Cyberbullying".
Random Forest Classifier: Displays similar results for comparison.

File Structure

.
├── app.py                 # Main Flask application
├── templates/
│   └── index.html         # Frontend HTML file
├── static/
│   ├── style.css          # CSS for styling
│   └── script.js          # JavaScript for async predictions
├── models/
│   ├── vectorizer.joblib  # Pre-trained vectorizer
│   ├── lr_model.joblib    # Logistic Regression model
│   └── rfc_model.joblib   # Random Forest model
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies

Example Comments for Testing
Comment	Expected Result
"You're such a loser."	Cyberbullying
"Have a great day, my friend!"	Not Cyberbullying
"Nobody likes you, just leave."	Cyberbullying
"Let's catch up this weekend!"	Not Cyberbullying


Future Improvements
Expand Models: Add more classifiers and compare their performances.
Multi-class Classification: Extend the application to identify subcategories of cyberbullying (e.g., sexual, racial, etc.).
Interactive Feedback: Allow users to report false predictions for model improvement.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
scikit-learn: For machine learning models and preprocessing tools.
spaCy: For text lemmatization.
nltk: For stopwords removal.
Datasets and resources for cyberbullying detection research.
markdown
Copy code

### Notes:
- Replace placeholders like `your-username` with your actual GitHub username or project details.
- Ensure the `requirements.txt` file includes all necessary dependencies for smooth setup
