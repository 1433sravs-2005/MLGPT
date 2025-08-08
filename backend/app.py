from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import json
import os

# ✅ Setup Flask app to serve React build
app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), '../client/build'), static_url_path='/')
CORS(app)

# ✅ Serve React index.html
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    file_path = os.path.join(app.static_folder, path)
    return send_from_directory(app.static_folder, path if path and os.path.exists(file_path) else 'index.html')

# ✅ ML Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    uploaded_file = request.files['file']
    algorithm = request.form.get('algorithm', 'decisiontree').lower()

    try:
        df = pd.read_csv(uploaded_file)

        if df.shape[1] < 2:
            return jsonify({'error': 'CSV must have at least 2 columns'})

        # Label encode categorical columns
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ✅ Supported algorithms
        algorithms = {
            'decisiontree': DecisionTreeClassifier(),
            'logisticregression': LogisticRegression(max_iter=1000),
            'randomforest': RandomForestClassifier(),
            'svm': SVC(),
            'knn': KNeighborsClassifier(),
            'naivebayes': GaussianNB()
        }

        if algorithm not in algorithms:
            return jsonify({'error': f"Unsupported algorithm '{algorithm}'. Choose from: {', '.join(algorithms.keys())}"})

        model = algorithms[algorithm]
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        # ✅ Python code to return
        code_snippet = f"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.{model.__module__.split('.')[-2]} import {model.__class__.__name__}

df = pd.read_csv("your_dataset.csv")

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = {model.__class__.__name__}()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
""".strip()

        return jsonify({
            'accuracy': round(accuracy * 100, 2),
            'python_code': code_snippet
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# ✅ Chatbot Endpoint
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        question = data.get("question", "").lower()

        responses_path = os.path.join(os.path.dirname(__file__), '../responses.json')
        with open(responses_path, "r") as f:
            responses = json.load(f)

        for keyword, reply in responses.items():
            if keyword in question:
                return jsonify({"answer": reply})

        return jsonify({"answer": "Sorry, I'm still learning. Try asking about ML models, pandas, or numpy!"})

    except Exception as e:
        print("Chat Error:", e)
        return jsonify({"answer": "Oops! Something went wrong with the chatbot."})

# ✅ Run server
if __name__ == '__main__':
    app.run(debug=True)
