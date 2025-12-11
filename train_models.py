import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz

DB_PATH = Path("recsys.db")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def load_data_from_db():
    if not DB_PATH.exists():
        raise FileNotFoundError("recsys.db not found. Run create_db.py first.")

    conn = sqlite3.connect(DB_PATH)
    movies = pd.read_sql_query("SELECT * FROM movies", conn)
    ratings = pd.read_sql_query("SELECT * FROM ratings", conn)
    conn.close()

    return movies, ratings


def train_collaborative_filtering_nmf(ratings, n_components=20):
    """
    Train matrix factorization model using NMF from scikit-learn.
    Ratings passed here are already SUBSETTED (small).
    """

    print("=== Training Collaborative Filtering model using NMF (no Surprise) ===")

    # Unique users and movies (only in the small subset)
    user_ids = sorted(ratings["userId"].unique())
    movie_ids = sorted(ratings["movieId"].unique())

    userId_to_index_cf = {uid: idx for idx, uid in enumerate(user_ids)}
    index_to_userId_cf = {idx: uid for idx, uid in enumerate(user_ids)}

    movieId_to_index_cf = {mid: idx for idx, mid in enumerate(movie_ids)}
    index_to_movieId_cf = {idx: mid for idx, mid in enumerate(movie_ids)}

    num_users = len(user_ids)
    num_movies = len(movie_ids)

    print(f"Number of users (subset): {num_users}, Number of movies (subset): {num_movies}")

    # Build dense user-item matrix (now small enough)
    R = np.zeros((num_users, num_movies), dtype=np.float32)

    for row in ratings.itertuples(index=False):
        u_idx = userId_to_index_cf[row.userId]
        i_idx = movieId_to_index_cf[row.movieId]
        R[u_idx, i_idx] = float(row.rating)

    nmf = NMF(
        n_components=n_components,
        init="random",
        random_state=42,
        max_iter=100,  # lighter
        solver="cd",
    )

    print("Fitting NMF model on reduced data...")
    user_factors = nmf.fit_transform(R)   # (num_users, k)
    item_factors = nmf.components_        # (k, num_movies)

    # Save NMF factors and mappings
    joblib.dump(user_factors, MODELS_DIR / "nmf_user_factors.pkl")
    joblib.dump(item_factors, MODELS_DIR / "nmf_item_factors.pkl")
    joblib.dump(userId_to_index_cf, MODELS_DIR / "userId_to_index_cf.pkl")
    joblib.dump(index_to_userId_cf, MODELS_DIR / "index_to_userId_cf.pkl")
    joblib.dump(movieId_to_index_cf, MODELS_DIR / "movieId_to_index_cf.pkl")
    joblib.dump(index_to_movieId_cf, MODELS_DIR / "index_to_movieId_cf.pkl")

    print("NMF model and CF mappings saved in models/ folder.")


def build_content_based_model(movies):
    """
    Build TF-IDF based content model using movie title + genres.
    Saves:
        - tfidf_vectorizer.pkl
        - movieId_to_index.pkl
        - index_to_movieId.pkl
        - content_cosine_sim.npz
    """
    print("=== Building Content-Based Model (TF-IDF on title+genres) ===")

    movies["genres"] = movies["genres"].fillna("")
    movies["title"] = movies["title"].fillna("")

    movies["text_features"] = movies["title"] + " " + movies["genres"]

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["text_features"])

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    movie_ids = movies["movieId"].values
    movieId_to_index_cb = {mid: idx for idx, mid in enumerate(movie_ids)}
    index_to_movieId_cb = {idx: mid for idx, mid in enumerate(movie_ids)}

    print("Computing cosine similarity matrix (may take a bit)...")
    cosine_sim = cosine_similarity(tfidf_matrix, dense_output=False)

    # Save artifacts
    joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(movieId_to_index_cb, MODELS_DIR / "movieId_to_index.pkl")
    joblib.dump(index_to_movieId_cb, MODELS_DIR / "index_to_movieId.pkl")
    save_npz(MODELS_DIR / "content_cosine_sim.npz", cosine_sim)

    print("Content-based artifacts saved in models/ folder.")


def main():
    print("Loading data from database...")
    movies, ratings = load_data_from_db()
    print(f"Movies: {movies.shape}, Ratings: {ratings.shape}")

    # 🔹 REDUCE DATA SIZE FOR FASTER TRAINING 🔹

    # 1) Keep only users with most ratings
    MAX_USERS = 500
    user_counts = ratings["userId"].value_counts()
    top_users = user_counts.head(MAX_USERS).index
    ratings_small = ratings[ratings["userId"].isin(top_users)]

    # 2) From those, keep only the movies with most ratings
    MAX_MOVIES = 1000
    movie_counts = ratings_small["movieId"].value_counts()
    top_movies = movie_counts.head(MAX_MOVIES).index
    ratings_small = ratings_small[ratings_small["movieId"].isin(top_movies)].reset_index(drop=True)

    print(f"After subsetting: Ratings: {ratings_small.shape}")

    # 1. Train CF with NMF on this reduced data
    train_collaborative_filtering_nmf(ratings_small)

    # 2. Content-Based model uses ALL movies (that’s fine)
    build_content_based_model(movies)

    # Save DataFrames for later use
    movies.to_pickle(MODELS_DIR / "movies_df.pkl")
    ratings.to_pickle(MODELS_DIR / "ratings_df.pkl")

    print("All models and data artifacts saved in models/ directory.")


if __name__ == "__main__":
    main()
