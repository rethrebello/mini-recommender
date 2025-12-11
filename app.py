import streamlit as st
from pathlib import Path

from recommender import HybridRecommender

MODELS_DIR = Path("models")

@st.cache_resource
def load_recommender():
    return HybridRecommender()

def main():
    st.set_page_config(page_title="Personalized Product Recommendation System",
                       layout="wide")

    st.title("🎯 Personalized Product Recommendation System (Mini Project)")
    st.write(
        """This mini-project demonstrates a **Hybrid Recommendation System** combining:
- **Collaborative Filtering (SVD / Matrix Factorization)**
- **Content-Based Filtering (TF-IDF on title + genres)**

Dataset: **MovieLens** (movie recommendation as a proxy for products).
"""
    )

    recommender = load_recommender()

    tab1, tab2 = st.tabs(["Existing User (Hybrid)", "New User (Content-Based)"])

    with tab1:
        st.subheader("Existing User Recommendations (Hybrid CF + Content-Based)")

        user_ids = sorted(recommender.ratings_df['userId'].unique().tolist())
        selected_user = st.selectbox("Select a User ID:", user_ids)

        n_recs = st.slider("Number of recommendations:", min_value=5, max_value=30, value=10, step=1)
        alpha = st.slider(
            "Hybrid weight (CF vs Content-based):",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="0 = only content-based, 1 = only collaborative filtering",
        )

        if st.button("Get Hybrid Recommendations"):
            with st.spinner("Generating recommendations..."):
                recs_df = recommender.recommend_for_user(
                    user_id=int(selected_user),
                    n=n_recs,
                    alpha=alpha,
                )
            st.success(f"Top {n_recs} recommendations for user {selected_user}")
            st.dataframe(recs_df)

            st.markdown("### User's top-rated movies")
            user_ratings = recommender.get_user_ratings(int(selected_user))
            user_top = (
                user_ratings.merge(recommender.movies_df, on="movieId")
                .sort_values("rating", ascending=False)
                .head(10)
            )
            st.dataframe(user_top[["movieId", "title", "genres", "rating"]])

    with tab2:
        st.subheader("New User (Cold Start) Recommendations")

        option = st.radio(
            "How to generate recommendations?",
            ("Show popular movies", "I will select a few movies I like"),
        )

        n_recs_new = st.slider(
            "Number of recommendations (new user):",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
            key="n_recs_new",
        )

        if option == "Show popular movies":
            if st.button("Recommend Popular Movies"):
                with st.spinner("Calculating popular movies..."):
                    recs_pop = recommender.recommend_for_new_user(n=n_recs_new)
                st.success("Popular movies (by rating and rating count):")
                st.dataframe(recs_pop)
        else:
            all_titles = sorted(recommender.movies_df["title"].unique().tolist())
            liked_titles = st.multiselect("Select some movies you like:", all_titles[:5000])

            if st.button("Recommend Similar Movies"):
                if not liked_titles:
                    st.warning("Please select at least one movie you like.")
                else:
                    with st.spinner("Generating content-based recommendations..."):
                        recs_cb = recommender.recommend_for_new_user_with_likes(
                            liked_titles, n=n_recs_new
                        )
                    st.success("Recommendations based on your liked movies:")
                    st.dataframe(recs_cb)

    st.markdown("---")
    st.write(
        """**Implementation Details**
- **Database:** SQLite (`recsys.db`) with `movies` and `ratings` tables.
- **Collaborative Filtering:** SVD (matrix factorization) using Surprise.
- **Content-Based:** TF-IDF on movie `title + genres` + cosine similarity.
- **Hybrid Score:** `alpha * CF_normalized + (1 - alpha) * Content_normalized`.
"""
    )

if __name__ == "__main__":
    main()
