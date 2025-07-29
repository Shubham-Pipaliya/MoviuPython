from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app import db
from app.recommendation_model import load_model, get_top_n, predict_rating, get_fallback_movies_for_user, get_movie_genre_vectors
from app.tv_show_recommender import load_show_model, get_top_n_shows, predict_show_rating, get_show_genre_vectors
from app.models import Movie, TVShow, User, Genre
from app.utils.mongo_data_loader import get_movies_df, get_shows_df, get_followings_df, get_reviews_df
from bson import ObjectId
import redis
import time
from datetime import timedelta
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime

app = FastAPI(title="Recommendation API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

followings_df = get_followings_df()
movies_df = get_movies_df()
shows_df = get_shows_df()

try:
    movie_model, movie_data, movie_reviews_df = load_model()
    show_model, show_data, show_reviews_df = load_show_model()
except Exception as e:
    print("Model loading failed:", e)
    raise

@app.get("/")
def serve_home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/health")
def health_check():
    return {"message": "Welcome to the Movie & TV Show Recommendation API"}

@app.get("/home/sections")
def get_home_sections(user_id: str = Query(...), language: str = Query("English")):
    print("[DEBUG] using hybrid recommender for home/sections")

    def get_user_genres(user_id: str):
        try:
            user = User.objects(id=ObjectId(user_id)).first()
            if not user or not user.genres:
                return []
            genre_objs = Genre.objects(id__in=user.genres).only("name")
            return [g.name for g in genre_objs if g.name]
        except Exception:
            return []

    def get_default_vector(preferred_genres, vector_df):
        default_vector = np.zeros((1, vector_df.shape[1]))
        for genre in preferred_genres:
            if genre in vector_df.columns:
                idx = vector_df.columns.get_loc(genre)
                default_vector[0][idx] = 1.0
        return default_vector

    def genre_based_recommendations(metas_df, vector_df):
        preferred_genres = get_user_genres(user_id)
        print("[✅] Mapped genre names:", preferred_genres)
        if not preferred_genres:
            return []
        default_vector = get_default_vector(preferred_genres, vector_df)
        candidates = metas_df[metas_df["language"].str.lower() == language.lower()].copy()
        candidate_ids = candidates["movie_id" if "movie_id" in candidates.columns else "show_id"].tolist()
        candidate_vectors = vector_df.loc[vector_df.index.intersection(candidate_ids)]
        if candidate_vectors.empty:
            return []
        similarities = cosine_similarity(default_vector, candidate_vectors.fillna(0).values)[0]
        candidates["similarity"] = similarities
        return (
            candidates.sort_values("similarity", ascending=False)
            .head(10)[[c for c in candidates.columns if c != "similarity"]]
            .to_dict(orient="records")
        )


    def fallback_items(df, key="movie_id"):
        return (
            df[df["language"].str.lower() == language.lower()][[key, "title", "genre", "language", "poster_url"]]
            .head(10)
            .to_dict(orient="records")
        )

    def get_recommended_movies():
        top_n = get_top_n(
            movie_model, movie_data, movie_reviews_df,
            n=10, hybrid=True,
            language_filter=language,
            metadata_df=movies_df
        )
        recs = top_n.get(user_id)
        if recs:
            movie_ids = [mid for mid, _ in recs]
            return movies_df[movies_df["movie_id"].isin(movie_ids)][["movie_id", "title", "genre", "language", "poster_url"]].to_dict(orient="records")
        genre_based = genre_based_recommendations(movies_df, get_movie_genre_vectors())
        return genre_based if genre_based else fallback_items(movies_df, key="movie_id")

    def get_recommended_shows():
        top_n = get_top_n_shows(
            show_model, show_data, show_reviews_df,
            n=10, hybrid=True,
            language_filter=language,
            metadata_df=shows_df
        )
        recs = top_n.get(user_id)
        if recs:
            show_ids = [sid for sid in recs]
            return shows_df[shows_df["show_id"].isin(show_ids)][["show_id", "title", "genre", "language", "poster_url"]].to_dict(orient="records")
        genre_based = genre_based_recommendations(shows_df, get_show_genre_vectors())
        return genre_based if genre_based else fallback_items(shows_df, key="show_id")

    def get_must_watch():
        agg = movie_reviews_df.groupby("movie").agg(avg_rating=("rating", "mean"), count=("rating", "count"))
        filtered = agg[(agg["avg_rating"] >= 4.5) & (agg["count"] >= 3)].reset_index()
        return movies_df[movies_df["movie_id"].isin(filtered["movie"])][
            ["movie_id", "title", "genre", "language", "poster_url"]
        ].to_dict(orient="records")


    def top_movies():
        top_movies = movies_df.sort_values("rating", ascending=False).head(10)
        return top_movies[["movie_id", "title", "genre", "language", "poster_url"]].to_dict(orient="records")

    def top_shows():
        top_shows = shows_df.sort_values("rating", ascending=False).head(10)
        return top_shows[["show_id", "title", "genre", "language", "poster_url"]].to_dict(orient="records")

    def coming_soon():
        if "release_date" not in movies_df.columns:
            print("[⚠] No 'release_date' column found in movies_df.")
            return []
        print(movies_df["release_date"].value_counts(dropna=False).head(10))
        # Convert release_date to datetime
        df = movies_df.copy()
        future_idx = movies_df.sample(10, random_state=42).index
        # Inject release dates 5 to 60 days in the future
        movies_df.loc[future_idx, "release_date"] = [
            (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(5, 65, 6)
        ]
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

        # Filter upcoming releases
        now = datetime.now()
        upcoming = df[
            (df["release_date"] > now) &
            (df["language"].str.lower() == language.lower())
        ].sort_values("release_date")

        if upcoming.empty:
            print("[ℹ] No upcoming movies found.")
            return []

        upcoming["poster_url"] = upcoming["poster_url"].fillna("https://example.com/default-poster.jpg")

        return upcoming[["movie_id", "title", "genre", "language", "poster_url"]].head(10).to_dict(orient="records")

    def newly_launched():
        # Placeholder for newly launched content
        if "release_date" not in shows_df.columns:
            print("[⚠] No 'release_date' column found in shows_df.")
            return []
        print(shows_df["release_date"].value_counts(dropna=False).head(10))
        # Convert release_date to datetime
        df = shows_df.copy()
        future_idx = shows_df.sample(10, random_state=42).index
        # Inject release dates 5 to 60 days in the future
        shows_df.loc[future_idx, "release_date"] = [
            (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(5, 65, 6)
        ]
        
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        # Filter newly launched shows
        now = datetime.now()
        newly_launched = df[
            (df["release_date"] > now) &
            (df["language"].str.lower() == language.lower())
        ].sort_values("release_date")
        if newly_launched.empty:
            print("[ℹ] No newly launched shows found.")
            return []
        newly_launched["poster_url"] = newly_launched["poster_url"].fillna("https://example.com/default-poster.jpg")
        return newly_launched[["show_id", "title", "genre", "language", "poster_url"]].head(10).to_dict(orient="records")

    return {
        "user_id": user_id,
        "trending_movies": get_recommended_movies(),
        "trending_shows": get_recommended_shows(),
        "must_watch": get_must_watch(),
        "top_movies": top_movies(),
        "top_shows": top_shows(),
        "coming_soon": coming_soon(),
        "newly_launched": newly_launched(),
        "trending_trailers": []
    }
