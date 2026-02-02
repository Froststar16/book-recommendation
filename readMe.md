# The Old Library: Book Recommendation System

A **semantic book recommendation system** using **Sentence Transformers** and **cosine similarity**, with a cozy, old-library-themed Flask UI. Discover books by themes, moods, or keywords, and get recommendations based on the content and metadata of 4,700+ popular books.

---

## 🌟 Features

- **Semantic Search**: Search by book themes, descriptions, or ideas.
- **Content + Metadata Fusion**: Uses book description, title, author, average rating, publication year, and language for recommendations.
- **Flask Web App**: Interactive, retro-style web interface inspired by classic libraries.
- **Cosine Similarity**: Lightweight, fast, and accurate recommendation algorithm.
- **Colab TPU Support**: Efficient embedding generation for large book datasets.

---

## 🛠️ Tech Stack

- **Python 3.14**
- **Flask**: Web framework for UI
- **Sentence Transformers**: `all-MiniLM-L6-v2` for semantic embeddings
- **Scikit-learn**: For cosine similarity and scaling numeric features
- **Pandas & NumPy**: Data handling and embedding manipulation
- **HTML / CSS**: Old-library aesthetic UI

---


---

## ⚠️ Note on Embeddings

The **embedding matrix (`embeddings.npy`)** generated in Colab TPU is **not included** in this repository to keep it lightweight.  

To generate embeddings locally or in Colab:

1. Open `notebooks/train_embeddings.ipynb`
2. Run the notebook (TPU recommended)
3. Save embeddings using:

```python
np.save("embeddings.npy", final_embeddings)


