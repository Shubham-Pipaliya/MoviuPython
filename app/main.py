from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from app import db
from app.recommendation_model import load_model, get_top_n, predict_rating, get_fallback_movies_for_user, get_movie_genre_vectors
from app.tv_show_recommender import load_show_model, get_top_n_shows, predict_show_rating, get_show_genre_vectors
from app.models import Movie, TVShow, User, Genre, MovieReview, ReviewShow
from app.utils.mongo_data_loader import get_movies_df, get_shows_df, get_followings_df, get_reviews_df
from app.utils.serializers import serialize_movies, serialize_shows
from bson import ObjectId
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import os

app = FastAPI(title="Recommendation API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

followings_df = get_followings_df()
movies_df = get_movies_df()
shows_df = get_shows_df()

# Load models
movie_model, movie_data, movie_reviews_df = load_model()
show_model, show_data, show_reviews_df = load_show_model()

@app.get("/home/sections")
def get_home_sections(user_id: str = Query(...), language: str = Query("English")):
    # Helper: get user preferred genres
    def get_user_genres(uid):
        user = User.objects(id=ObjectId(uid)).first()
        if not user or not user.genres:
            return []
        genre_ids = [str(g) for g in user.genres]
        return [g.name for g in Genre.objects(id__in=genre_ids).only("name")]

    # Genre-based fallback to get IDs
    def genre_based_ids(meta_df, vector_df, key):
        prefs = get_user_genres(user_id)
        if not prefs:
            return []
        default_vec = np.zeros((1, vector_df.shape[1]))
        for g in prefs:
            if g in vector_df.columns:
                default_vec[0, vector_df.columns.get_loc(g)] = 1.0
        candidates = meta_df[meta_df["language"].str.lower() == language.lower()]
        ids = candidates[key].tolist()
        vecs = vector_df.loc[vector_df.index.intersection(ids)]
        if vecs.empty:
            return []
        sims = cosine_similarity(default_vec, vecs.fillna(0).values)[0]
        ranked = vecs.assign(similarity=sims).sort_values("similarity", ascending=False)
        return ranked.head(10).index.tolist()

    # Generic fallback IDs
    def fallback_ids(meta_df, key):
        return meta_df[meta_df["language"].str.lower() == language.lower()][key].head(10).tolist()

    # Movie Sections
    def get_recommended_movies():
        recs = get_top_n(
            movie_model, movie_data, movie_reviews_df,
            n=10, hybrid=True, language_filter=language,
            metadata_df=movies_df
        ).get(user_id, [])
        if recs:
            ids = [mid for mid, _ in recs]
        else:
            ids = genre_based_ids(movies_df, get_movie_genre_vectors(), "movie_id")
            if not ids:
                ids = fallback_ids(movies_df, "movie_id")
        return serialize_movies(ids)

    def get_recommended_shows():
        recs = get_top_n_shows(
            show_model, show_data, show_reviews_df,
            n=10, hybrid=True, language_filter=language,
            metadata_df=shows_df
        ).get(user_id, [])
        if recs:
            ids = recs
        else:
            ids = genre_based_ids(shows_df, get_show_genre_vectors(), "show_id")
            if not ids:
                ids = fallback_ids(shows_df, "show_id")
        return serialize_shows(ids)

    def get_must_watch():
        pipeline = [
            {"$match": {"is_deleted": False}},
            {"$group": {"_id": "$movie", "avgRating": {"$avg": "$rating"}, "reviewCount": {"$sum": 1}}},
            {"$match": {"avgRating": {"$gte": 4.5}, "reviewCount": {"$gte": 3}}}
        ]
        stats = list(MovieReview.objects.aggregate(*pipeline))
        ids = [s["_id"] for s in stats]
        return serialize_movies(ids)

    def top_movies():
        ids = movies_df.sort_values("rating", ascending=False).head(10)["movie_id"].tolist()
        return serialize_movies(ids)

    def top_shows():
        ids = shows_df.sort_values("rating", ascending=False).head(10)["show_id"].tolist()
        return serialize_shows(ids)

    def coming_soon():
        df = movies_df.copy()
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        upcoming = df[(df["release_date"] > datetime.now()) & (df["language"].str.lower() == language.lower())]
        ids = upcoming.sort_values("release_date").head(10)["movie_id"].tolist()
        return serialize_movies(ids)

    def newly_launched():
        df = shows_df.copy()
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        recent = df[(df["release_date"] > datetime.now()) & (df["language"].str.lower() == language.lower())]
        ids = recent.sort_values("release_date").head(10)["show_id"].tolist()
        return serialize_shows(ids)

    # Build result
    result = {
        "user_id": user_id,
        "trending_movies": get_recommended_movies(),
        "trending_shows": get_recommended_shows(),
        "must_watch": get_must_watch(),
        "top_movies": top_movies(),
        "top_shows": top_shows(),
        "coming_soon": coming_soon(),
        "newly_launched": newly_launched()
    }

    # Encode and return pure JSON with ObjectId → str
    payload = jsonable_encoder(
        result,
        custom_encoder={ ObjectId: str }
    )
    return JSONResponse(content=payload)
