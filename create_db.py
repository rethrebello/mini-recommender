import sqlite3
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DB_PATH = Path("recsys.db")

def create_database():
    movies_path = DATA_DIR / "movies.csv"
    ratings_path = DATA_DIR / "ratings.csv"

    if not movies_path.exists() or not ratings_path.exists():
        raise FileNotFoundError("movies.csv or ratings.csv not found in data/ folder.")

    print("Loading CSV files...")
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    print("Creating SQLite database recsys.db...")
    conn = sqlite3.connect(DB_PATH)

    movies.to_sql("movies", conn, if_exists="replace", index=False)
    ratings.to_sql("ratings", conn, if_exists="replace", index=False)

    # Simple indexes to speed up lookups
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(userId);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movieId);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_movies_movieid ON movies(movieId);")

    conn.commit()
    conn.close()
    print("Database created successfully at recsys.db")

if __name__ == "__main__":
    create_database()
