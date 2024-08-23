from flask import Flask, request, jsonify
import nltk

app = Flask(__name__)

@app.route("/")

def chat():
    return "hi"

if __name__ == "__main__":
    app.run(debug=True)
