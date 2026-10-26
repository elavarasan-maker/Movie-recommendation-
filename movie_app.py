"""
Movie Recommendation System - beginner version (content-based filtering)
Run with:  streamlit run movie_app.py
"""
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- 1. DATA ----------
# Small demo dataset. For a real resume project, replace it with the free
# MovieLens dataset (movies.csv has title + genres) and load it with pd.read_csv().
movies = pd.DataFrame({
    "title": [
        "The Dark Knight", "Batman Begins", "Inception", "Interstellar",
        "The Matrix", "Toy Story", "Finding Nemo", "Shrek",
        "The Godfather", "Goodfellas", "Titanic", "The Notebook",
        "Avengers: Endgame", "Iron Man", "The Hangover", "Superbad",
        "Get Out", "The Conjuring", "Jurassic Park", "Avatar",
    ],
    "genres": [
        "action crime drama", "action crime drama", "action sci-fi thriller", "adventure drama sci-fi",
        "action sci-fi", "animation comedy family", "animation adventure family", "animation comedy family",
        "crime drama", "crime drama biography", "drama romance", "drama romance",
        "action adventure sci-fi", "action adventure sci-fi", "comedy", "comedy",
        "horror thriller mystery", "horror thriller", "adventure sci-fi thriller", "action adventure sci-fi",
    ],
})

# ---------- 2. BUILD THE MODEL ----------
# Turn each movie's genres into numbers, then measure how similar movies are.
@st.cache_resource
def build_similarity():
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(movies["genres"])
    return cosine_similarity(matrix)

similarity = build_similarity()

def recommend(title, n=5):
    index = movies[movies["title"] == title].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != index][:n]   # skip the movie itself
    return [(movies.loc[i, "title"], movies.loc[i, "genres"], round(score * 100)) for i, score in scores]

# ---------- 3. THE APP ----------
st.title("🎬 Movie Recommendation System")
st.write("Pick a movie you like and get similar ones.")

choice = st.selectbox("Choose a movie:", movies["title"])
count = st.slider("How many recommendations?", 1, 10, 5)

if st.button("Recommend"):
    for title, genres, score in recommend(choice, count):
        st.subheader(title)
        st.caption(f"Genres: {genres} | Match: {score}%")

st.caption("Method: TF-IDF + cosine similarity (content-based filtering)")
