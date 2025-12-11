from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from scipy.sparse import load_npz

MODELS_DIR = Path("models")


class HybridRecommender:
    def __init__(self):
        print("Loading trained models and data (NMF + Content-based)...")

        # === Collaborative Filtering artifacts (NMF) ===
        self.user_factors = joblib.load(MODELS_DIR / "nmf_user_factors.pkl")   # (num_users, k)
        self.item_factors = joblib.load(MODELS_DIR / "nmf_item_factors.pkl")   # (k, num_movies)
        self.userId_to_index_cf = joblib.load(MODELS_DIR / "userId_to_index_cf.pkl")
        self.index_to_userId_cf = joblib.load(MODELS_DIR / "index_to_userId_cf.pkl")
        self.movieId_to_index_cf = joblib.load(MODELS_DIR / "movieId_to_index_cf.pkl")
        self.index_to_movieId_cf = joblib.load(MODELS_DIR / "index_to_movieId_cf.pkl")

        # === Content-Based artifacts ===
        # (We don't strictly need tfidf_vectorizer at prediction time, but we load mappings + similarities)
        self.movieId_to_index_cb = joblib.load(MODELS_DIR / "movieId_to_index.pkl")
        self.index_to_movieId_cb = joblib.load(MODELS_DIR / "index_to_movieId.pkl")
        self.cosine_sim = load_npz(MODELS_DIR / "content_cosine_sim.npz")

        # DataFrames
        self.movies_df: pd.DataFrame = pd.read_pickle(MODELS_DIR / "movies_df.pkl")
        self.ratings_df: pd.DataFrame = pd.read_pickle(MODELS_DIR / "ratings_df.pkl")

        self.all_movie_ids = self.movies_df["movieId"].unique()

    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        return self.ratings_df[self.ratings_df["userId"] == user_id]

    def _normalize_scores(self, scores_dict):
        if not scores_dict:
            return scores_dict

        values = np.array(list(scores_dict.values()))
        min_v, max_v = values.min(), values.max()
        if max_v - min_v == 0:
            return {k: 0.5 for k in scores_dict}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

    def _predict_rating_cf(self, user_id: int, movie_id: int) -> float:
        """Predict rating via NMF factors; if unknown user/movie, return 0."""
        if user_id not in self.userId_to_index_cf:
            return 0.0
        if movie_id not in self.movieId_to_index_cf:
            return 0.0

        u_idx = self.userId_to_index_cf[user_id]
        m_idx = self.movieId_to_index_cf[movie_id]

        # user_factors[u_idx] shape: (k,)
        # item_factors[:, m_idx] shape: (k,)
        return float(np.dot(self.user_factors[u_idx, :], self.item_factors[:, m_idx]))

    def recommend_for_user(
        self,
        user_id: int,
        n: int = 10,
        alpha: float = 0.6,
        min_rating_for_like: float = 4.0,
    ):
        """
        Hybrid recommendations for existing user:
        - Collaborative Filtering using NMF
        - Content-based similarity using TF-IDF/cosine_sim
        alpha = weight for CF component (0..1)
        """

        user_ratings = self.get_user_ratings(user_id)

        # If user has no ratings → cold start: use popularity
        if user_ratings.empty or user_id not in self.userId_to_index_cf:
            print("User is new or not in CF model. Using popularity-based recs.")
            return self.recommend_for_new_user(n=n)

        rated_movie_ids = set(user_ratings["movieId"].tolist())

        # Only movies present in CF mapping AND not already rated
        candidate_movie_ids = [
            mid
            for mid in self.all_movie_ids
            if (mid in self.movieId_to_index_cf and mid not in rated_movie_ids)
        ]

        # 1. CF scores via NMF factors
        cf_scores = {}
        for mid in candidate_movie_ids:
            cf_scores[mid] = self._predict_rating_cf(user_id, mid)

        # 2. Content-based scores: similarity to liked movies (rating >= min_rating_for_like)
        liked = user_ratings[user_ratings["rating"] >= min_rating_for_like]
        liked_movie_ids = liked["movieId"].tolist()

        cb_scores = {}
        if liked_movie_ids:
            liked_indices_cb = [
                self.movieId_to_index_cb[mid]
                for mid in liked_movie_ids
                if mid in self.movieId_to_index_cb
            ]

            for mid in candidate_movie_ids:
                if mid not in self.movieId_to_index_cb:
                    continue
                idx_cb = self.movieId_to_index_cb[mid]
                sim_values = self.cosine_sim[idx_cb, liked_indices_cb].toarray().flatten()
                if len(sim_values) == 0:
                    cb_scores[mid] = 0.0
                else:
                    cb_scores[mid] = float(np.mean(sim_values))
        else:
            # If no high ratings, default content score = 0 for all
            cb_scores = {mid: 0.0 for mid in candidate_movie_ids}

        # Normalize CF and CB scores
        cf_norm = self._normalize_scores(cf_scores)
        cb_norm = self._normalize_scores(cb_scores)

        # 3. Hybrid aggregation
        final_scores = {}
        for mid in candidate_movie_ids:
            cf = cf_norm.get(mid, 0.0)
            cb = cb_norm.get(mid, 0.0)
            final_scores[mid] = alpha * cf + (1 - alpha) * cb

        # Top N results
        top_movies = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        top_movie_ids = [mid for mid, _ in top_movies]

        result_df = self.movies_df[self.movies_df["movieId"].isin(top_movie_ids)].copy()
        score_map = dict(top_movies)
        result_df["hybrid_score"] = result_df["movieId"].map(score_map)
        result_df["pred_rating_cf"] = result_df["movieId"].apply(
            lambda mid: cf_scores.get(mid, np.nan)
        )

        result_df = result_df.sort_values("hybrid_score", ascending=False)

        return result_df[["movieId", "title", "genres", "pred_rating_cf", "hybrid_score"]]

    def recommend_for_new_user(self, n: int = 10):
        """
        Popularity-based recommendation for brand new users.
        """
        movie_stats = (
            self.ratings_df.groupby("movieId")["rating"]
            .agg(["mean", "count"])
            .reset_index()
        )
        movie_stats["score"] = movie_stats["mean"] * np.log1p(movie_stats["count"])
        top = movie_stats.sort_values("score", ascending=False).head(n)
        result = top.merge(self.movies_df, on="movieId", how="left")
        return result[["movieId", "title", "genres", "mean", "count", "score"]]

    def recommend_for_new_user_with_likes(self, liked_titles, n: int = 10):
        """
        For a new user who selects some liked movies by title:
        Content-based recommendations based on similarity to those movies.
        """
        if not liked_titles:
            return self.recommend_for_new_user(n=n)

        liked_movies = self.movies_df[self.movies_df["title"].isin(liked_titles)]
        liked_indices_cb = [
            self.movieId_to_index_cb[mid]
            for mid in liked_movies["movieId"].tolist()
            if mid in self.movieId_to_index_cb
        ]

        if not liked_indices_cb:
            return self.recommend_for_new_user(n=n)

        # Average similarity across liked movies
        sim_vector = np.array(self.cosine_sim[liked_indices_cb].mean(axis=0)).flatten()

        # Get indices of top movies
        top_indices = sim_vector.argsort()[::-1][: n + len(liked_indices_cb)]

        liked_set = set(liked_indices_cb)
        final_indices = [idx for idx in top_indices if idx not in liked_set][:n]

        movie_ids = [self.index_to_movieId_cb[idx] for idx in final_indices]
        scores = [sim_vector[idx] for idx in final_indices]

        result = self.movies_df[self.movies_df["movieId"].isin(movie_ids)].copy()
        score_map = dict(zip(movie_ids, scores))
        result["content_score"] = result["movieId"].map(score_map)
        result = result.sort_values("content_score", ascending=False)

        return result[["movieId", "title", "genres", "content_score"]]
