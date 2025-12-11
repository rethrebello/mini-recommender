Mini Project: Personalized Product (Movie) Recommendation System

Steps to run:

1. Create a virtual environment (optional but recommended)
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate

2. Install dependencies:
   pip install -r requirements.txt

3. Download MovieLens dataset (e.g., ml-latest-small) from GroupLens.
   Copy 'movies.csv' and 'ratings.csv' into the 'data' folder in this project.

4. Create the SQLite database:
   python create_db.py

5. Train the models:
   python train_models.py

6. Run the Streamlit app:
   streamlit run app.py

Then open the link shown in the terminal (usually http://localhost:8501).
