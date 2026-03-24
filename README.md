# 🎬 Hybrid Movie Recommender System

> A hybrid recommendation engine combining **Collaborative Filtering (NMF)** and **Content-Based Filtering (TF-IDF + Cosine Similarity)** — built with Python, SQLite, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square&logo=streamlit)
![SQLite](https://img.shields.io/badge/SQLite-Database-lightgrey?style=flat-square&logo=sqlite)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#️-installation--setup)
- [Usage](#-usage)
- [Output](#-output)
- [Screenshots](#-screenshots)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔀 Hybrid Recommendations | Combines CF + Content-Based scores with tunable weight |
| 🎚️ Adjustable Alpha | Slide between collaborative and content-based filtering |
| 👤 Existing Users | Personalized recommendations from historical ratings |
| 🆕 Cold-Start Handling | Popularity-based or genre-based suggestions for new users |
| 🎯 Content-Based Mode | Recommendations from user-selected liked movies |
| 🌐 Interactive UI | Clean, browser-based interface powered by Streamlit |

---

## 🧠 How It Works

### 1. Data Pipeline (`create_db.py`)

1. Reads `movies.csv` and `ratings.csv` using **pandas**
2. Inserts into a **SQLite** database (`recsys.db`) with two tables:
   - `movies(movieId, title, genres, ...)`
   - `ratings(userId, movieId, rating, timestamp)`
3. Adds indexes on key columns for fast query performance

---

### 2. Model Training (`train_models.py`)

#### Collaborative Filtering (CF) — NMF

1. Filters to **top 500 users** and **top 1000 movies** to keep the matrix manageable
2. Builds a dense **user × item rating matrix** `R`
3. Trains `sklearn.decomposition.NMF` to decompose `R` into:
   - `user_factors` — shape `(num_users, k)`
   - `item_factors` — shape `(k, num_movies)`
4. Saves `nmf_user_factors.pkl`, `nmf_item_factors.pkl`, and ID ↔ index mapping dicts

#### Content-Based Filtering (CB) — TF-IDF

1. Combines each movie's **title + genres** into a text field
2. Applies **TF-IDF vectorization** across all movies
3. Computes a full pairwise **cosine similarity matrix**
4. Saves `tfidf_vectorizer.pkl`, `content_cosine_sim.npz`, and mapping dicts

#### Artifacts also saved: `movies_df.pkl`, `ratings_df.pkl`

---

### 3. Recommendation Logic (`recommender.py`)

#### Existing User → Hybrid Recommendations

**CF Score** — predicted rating via NMF dot product:
```
CF Score = user_factors[u] · item_factors[:, m]
```

**CB Score** — for each candidate movie, averages cosine similarity against all movies the user has rated ≥ 4 stars, then normalizes to `[0, 1]`.

Both scores are normalized and blended:

```
Hybrid Score = α × CF Score + (1 - α) × CB Score
```

- `α = 1.0` → Pure Collaborative Filtering  
- `α = 0.0` → Pure Content-Based Filtering  
- `α = 0.5` → Equal blend (recommended default)

#### New User → Cold-Start Recommendations

| Method | How it works |
|---|---|
| **Popularity-based** | Score = `mean_rating × log(rating_count)` — surfaces well-rated and widely seen films |
| **Liked movies** | User picks titles they enjoyed → CB cosine similarity finds the closest matches |

---

## 🏗️ Project Structure

```
mini-recommender/
│
├── data/
│   ├── movies.csv
│   └── ratings.csv
│
├── models/
│   ├── nmf_user_factors.pkl
│   ├── nmf_item_factors.pkl
│   ├── tfidf_vectorizer.pkl
│   └── content_cosine_sim.npz
│
├── create_db.py        # Sets up SQLite database
├── train_models.py     # Trains NMF and TF-IDF models
├── recommender.py      # Core recommendation logic
├── app.py              # Streamlit frontend
├── recsys.db           # SQLite database (auto-generated)
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python 3.8+ |
| Database | SQLite |
| ML / Math | scikit-learn, scipy, numpy |
| Data | pandas |
| Serialization | joblib |

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- [MovieLens dataset](https://grouplens.org/datasets/movielens/) (`movies.csv` and `ratings.csv`)

---

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/mini-recommender.git
cd mini-recommender
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Dataset

Place the MovieLens CSV files in the `data/` directory:

```
data/
├── movies.csv
└── ratings.csv
```

### 4. Create the Database

```bash
python create_db.py
```

### 5. Train the Models

```bash
python train_models.py
```

### 6. Launch the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 🖥️ Usage

### Tab 1 — Existing User

1. Select a **User ID** from the dropdown
2. Choose **N** (number of recommendations) and adjust the **alpha slider**
3. Get **hybrid recommendations** with CF + CB scores
4. See the user's own **top-rated movies** alongside results

### Tab 2 — New User (Cold Start)

**Option A — Popularity-based:**  
Recommends widely-seen, well-rated movies using `mean_rating × log(count)` scoring.

**Option B — Liked movies:**  
Pick titles you've enjoyed from a dropdown → content-based similarity returns the closest matches.

---

## 📊 Output

Each recommendation includes:

| Field | Description |
|---|---|
| 🎬 Movie Title | Recommended movie name |
| ⭐ CF Score | Predicted rating from collaborative filtering |
| 🔀 Hybrid Score | Weighted combination of CF + CB scores |
| ❤️ Liked Movies | User's historically rated movies (existing users) |

---

## 📸 Screenshots

<img width="1917" height="1032" alt="Screenshot (752)" src="https://github.com/user-attachments/assets/d1fa2225-d83c-4a68-bd91-9b8de44fa263" />
<img width="1920" height="1025" alt="Screenshot (754)" src="https://github.com/user-attachments/assets/eb71d833-d3b6-4a02-841d-3113e9b93066" />
<img width="1920" height="1037" alt="Screenshot (757)" src="https://github.com/user-attachments/assets/75ca661c-dd25-4ddb-a81d-65eb0d0b4a96" />


---

## 💡 Future Improvements

- [ ] Real-time user feedback loop for online learning
- [ ] Neural Collaborative Filtering (NCF / deep learning)
- [ ] Sparse matrix handling + ALS for larger datasets (e.g. `implicit` library)
- [ ] Replace NMF truncation with `Surprise` library for better CF
- [ ] REST API with FastAPI or Flask
- [ ] Explicit model retrain path within the Streamlit app
- [ ] Error handling for missing or corrupt model files
- [ ] Docker containerization & cloud deployment (AWS / GCP / Heroku)

---

## 👨‍💻 Author

**Reth Rebello**  
Engineering Student · ML Enthusiast · Data Analytics · software Developer

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
If you found this useful, give it a ⭐ on GitHub!
