from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data
df = pd.read_csv("model/metadata.csv")
embeddings = np.load("model/embeddings.npy")

model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

def recommend_from_text(query, top_n=10):
    query = clean_text(query)
    query_embedding = model.encode([query])

    query_embedding = np.hstack([
        query_embedding,
        np.zeros((1, embeddings.shape[1] - query_embedding.shape[1]))
    ])

    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    sims = cosine_similarity(query_embedding, embeddings).flatten()
    top_indices = sims.argsort()[::-1][:top_n]

    return df.iloc[top_indices]

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    query = ""

    if request.method == "POST":
        query = request.form["query"]
        recommendations = recommend_from_text(query)

    return render_template(
        "index.html",
        recommendations=recommendations,
        query=query
    )

if __name__ == "__main__":
    app.run(debug=True)
