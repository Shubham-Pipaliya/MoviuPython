from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app import db
from app.tv_show_recommender import load_show_model, get_top_n_shows, predict_show_rating
from app.utils.mongo_data_loader import get_movies_df, get_shows_df, get_followings_df
# from app.utils.mongo_data_loader import get_review_shows_df
from app.utils.mongo_data_loader import get_review_shows_df
from bson import ObjectId
from fastapi import APIRouter, Query
from app.utils.mongo_data_loader import get_movies_df, get_shows_df
from app.tv_show_recommender import get_top_n_shows, predict_show_rating
from app.models import Genre
import redis
import time
import os
import numpy as np
import pandas as pd
from app.recommendation_model import load_model, get_top_n, predict_rating, get_fallback_movies_for_user, get_movie_genre_vectors
from app.models import Movie, TVShow, User
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Recommendation API")

api_v1 = APIRouter(prefix="/api/v1")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

followings_df = get_followings_df()

GENRE_CSV_PATH = os.path.join(BASE_DIR, "..", "data", "SpakMobileApi.genres.csv")
genre_df = pd.read_csv(GENRE_CSV_PATH)
# Normalize all _id values whether they are raw or wrapped
genre_df["_id"] = genre_df["_id"].astype(str).str.extract(r'([a-f0-9]{24})')
genre_lookup = dict(zip(genre_df["_id"], genre_df["name"]))


def get_user_genres(user_id: str):
    try:
        user = User.objects(id=ObjectId(user_id)).first()
    except Exception as e:
        print(f"[❌] Invalid user_id format: {user_id} → {e}")
        return []

    if not user:
        print(f"[❌] User not found for _id: {user_id}")
        return []

    if not user.genres:
        print(f"[⚠] User {user_id} found but has no genres.")
        return []

    genre_ids = [str(g) for g in user.genres]
    matched = [genre_lookup.get(gid) for gid in genre_ids if gid in genre_lookup]
    print(f"[✅] Mapped genre names: {matched}")
    return matched

@app.get("/")
def serve_home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# --- Load models at startup ---
try:
    movie_model, movie_data, movie_reviews_df = load_model()
    assert get_movie_genre_vectors() is not None, "❌ movie_genre_vectors failed to initialize"
    show_model, show_data, show_reviews_df = load_show_model()
except Exception as e:
    print("Model loading failed:", e)
    raise
# --- Health Check ---
@app.get("/health")
def read_root():
    return {"message": "Welcome to the Movie & TV Show Recommendation API"}

# --- Movie Recommendation ---
movies_df = get_movies_df()  # metadata

@app.get("/recommend/movies/{language}/{user_id}")
def recommend_movies_by_language(language: str, user_id: str, n: int = 10):
    top_n_lang = get_top_n(
        movie_model,
        movie_data,
        movie_reviews_df,
        n=n,
        hybrid=True,
        language_filter=language,
        metadata_df=movies_df
    )

    recs = top_n_lang.get(user_id)

    if not recs:
        trending = movies_df[movies_df['language'].str.lower() == language.lower()]
        trending_data = trending[['movie_id', 'title']].head(n).to_dict(orient='records')
        return {
            "language": language,
            "user_id": user_id,
            "recommendations": trending_data
        }

    movie_ids = [mid for mid, _ in recs]
    movie_names = movies_df[movies_df['movie_id'].isin(movie_ids)][['movie_id', 'title']].to_dict(orient='records')

    return {
        "language": language,
        "user_id": user_id,
        "recommendations": movie_names
    }

@app.get("/predict/movie")
def predict_movie(user_id: str, movie_id: str):
    rating = predict_rating(movie_model, user_id, movie_id)
    return {"user_id": user_id, "movie_id": movie_id, "predicted_rating": rating}

# --- TV Show Recommendation ---
@app.get("/recommend/shows/{language}/{user_id}")
def recommend_shows_by_language(language: str, user_id: str, n: int = 10):
    top_n_lang = get_top_n_shows(
        show_model, show_data, show_reviews_df,
        n=n, hybrid=True,
        language_filter=language,
        metadata_df=get_shows_df()
    )

    recs = top_n_lang.get(user_id)
    if not recs:
        raise HTTPException(status_code=404, detail=f"No {language.title()} TV show recommendations found.")

    return {
        "language": language,
        "user_id": user_id,
        "recommendations": [sid for sid, _ in recs]
    }
@app.get("/predict/show")
def predict_show(user_id: str, show_id: str):
    rating = predict_show_rating(show_model, user_id, show_id)
    return {"user_id": user_id, "show_id": show_id, "predicted_rating": rating}

# --- Redis setup ---
redis_client = redis.Redis(
    host='127.0.0.1',
    port=6379,
    decode_responses=True
)

@app.get("/RedisTimingTest")
def test_redis():
    start = time.time()

    redis_client.set('test_key1', 'hello1')
    redis_client.set('test_key2', 'hello2')
    redis_client.set('test_key3', 'hello3')
    redis_client.set('test_key4', 'hello4')
    redis_client.set('test_key5', 'hello5')

    value1 = redis_client.get('test_key1')
    value2 = redis_client.get('test_key2')
    value3 = redis_client.get('test_key3')
    value4 = redis_client.get('test_key4')
    value5 = redis_client.get('test_key5')

    duration = time.time() - start
    return {
        "test_key1": value1,
        "test_key2": value2,
        "test_key3": value3,
        "test_key4": value4,
        "test_key5": value5,
        "duration_in_seconds": round(duration, 6)
    }


@api_v1.get("/recommendations")
def get_recommendations(
    user_id: str = Query(...),
    type: str = Query(..., regex="^(movie|show)$"),
    language: str = Query(...),
    n: int = 10
):
    if type == "movie":
        # Try full model recommendation first
        top_n = get_top_n(
            movie_model, movie_data, movie_reviews_df,
            n=n, hybrid=True,
            language_filter=language,
            metadata_df=movies_df
        )
        recs = top_n.get(user_id)

        # If model fails to return recs
        if not recs:
            preferred_genres = get_user_genres(user_id)

            if preferred_genres:
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np

                default_vector = np.zeros((1, get_movie_genre_vectors().shape[1]))
                for genre in preferred_genres:
                    if genre in get_movie_genre_vectors().columns:
                        default_vector[0][get_movie_genre_vectors().columns.get_loc(genre)] = 1.0

                candidate_movies = movies_df[
                    movies_df["language"].str.lower() == language.lower()
                ].copy()
                candidate_ids = candidate_movies["movie_id"].tolist()
                candidate_vectors = get_movie_genre_vectors().loc[
                    get_movie_genre_vectors().index.intersection(candidate_ids)
                ]
                if not candidate_vectors.empty:
                    similarities = cosine_similarity(default_vector, candidate_vectors.fillna(0).values)[0]
                    candidate_movies["similarity"] = similarities
                    top_matches = candidate_movies.sort_values("similarity", ascending=False).head(n)

                    return {
                        "user_id": user_id,
                        "type": "movie",
                        "recommendations": top_matches[["movie_id", "title", "genre", "language", "poster_url"]].to_dict(orient="records")
                    }


            # 🔁 Otherwise fallback to followings/ratings
            fallback = get_fallback_movies_for_user(
                user_id=user_id,
                language=language,
                movie_reviews_df=movie_reviews_df,
                followings_df=followings_df,
                movies_df=movies_df,
                n=n
            )
            if fallback:
                return {
                    "user_id": user_id,
                    "type": "movie",
                    "recommendations": fallback
                }

            # 🔚 Last resort: trending
            fallback = (
                movies_df[movies_df["language"].str.lower() == language.lower()]
                [["movie_id", "title", "genre", "language", "poster_url"]]
                .head(n)
                .to_dict(orient="records")
            )
            return {
                "user_id": user_id,
                "type": "movie",
                "recommendations": fallback
            }

        # ✅ Standard flow if recs exist
        movie_ids = [mid for mid, _ in recs]
        rec_data = movies_df[movies_df["movie_id"].isin(movie_ids)]
        return {
            "user_id": user_id,
            "type": "movie",
            "recommendations": rec_data[["movie_id", "title", "genre", "language", "poster_url"]].to_dict(orient="records")
        }

    elif type == "show":
        shows_df = get_shows_df()
        top_n = get_top_n_shows(
            show_model, show_data, show_reviews_df,
            n=n, hybrid=True,
            language_filter=language,
            metadata_df=shows_df
        )
        recs = top_n.get(user_id)

        if not recs:
            fallback = shows_df[shows_df["language"].str.lower() == language.lower()]
            return {
                "user_id": user_id,
                "type": "show",
                "recommendations": fallback[["show_id", "title", "genre", "language", "poster_url"]].head(n).to_dict(orient="records")
            }

        show_ids = [sid for sid in recs]
        rec_data = shows_df[shows_df["show_id"].isin(show_ids)]

        return {
            "user_id": user_id,
            "type": "show",
            "recommendations": rec_data[["show_id", "title", "genre", "language", "poster_url"]].to_dict(orient="records")
        }

followings_df = get_followings_df()
movies_df = get_movies_df()
shows_df = get_shows_df()
# show_reviews_df = get_review_shows_df()


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

    def get_default_vector(preferred_genres):
        default_vector = np.zeros((1, get_movie_genre_vectors().shape[1]))
        for genre in preferred_genres:
            if genre in get_movie_genre_vectors().columns:
                idx = get_movie_genre_vectors().columns.get_loc(genre)
                default_vector[0][idx] = 1.0
        return default_vector

    def genre_based_recommendations():
        preferred_genres = get_user_genres(user_id)
        print("[✅] Mapped genre names:", preferred_genres)
        if not preferred_genres:
            return []
        default_vector = get_default_vector(preferred_genres)
        candidate_movies = movies_df[movies_df["language"].str.lower() == language.lower()].copy()
        candidate_ids = candidate_movies["movie_id"].tolist()
        candidate_vectors = get_movie_genre_vectors().loc[get_movie_genre_vectors().index.intersection(candidate_ids)]
        if candidate_vectors.empty:
            return []
        similarities = cosine_similarity(default_vector, candidate_vectors.fillna(0).values)[0]
        candidate_movies["similarity"] = similarities
        return candidate_movies.sort_values("similarity", ascending=False).head(10).to_dict(orient="records")

    def fallback_movies():
        return (
            movies_df[movies_df["language"].str.lower() == language.lower()][["movie_id", "title", "genre", "language", "poster_url"]]
            .head(10)
            .to_dict(orient="records")
        )

    def fallback_show():
        return (
            shows_df[shows_df["language"].str.lower() == language.lower()][["show_id", "title", "genre", "language", "poster_url"]]
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
        genre_based = genre_based_recommendations()
        return genre_based if genre_based else fallback_movies()

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
        genre_based = genre_based_recommendations()
        return genre_based if genre_based else fallback_show(shows_df, key="show_id")
    return {
        "user_id": user_id,
        "trending_movies": get_recommended_movies(),
        "trending_shows": get_recommended_shows(),
        "must_watch": [],
        "top_movies": [],
        "top_shows": [],
        "coming_soon": [],
        "newly_launched": [],
        "trending_trailers": []
    }
   

print(genre_df.head(1))
print("User genre names:", get_user_genres("67977895dee8c7cd3de1d9bb"))
print("BASE_DIR:", BASE_DIR)
print("STATIC_DIR:", STATIC_DIR)
print("Static Files:", os.listdir(STATIC_DIR))
print("Genre vector columns:", list(get_movie_genre_vectors().columns))
print("[✅] movie_genre_vectors shape:", get_movie_genre_vectors().shape)

app.include_router(api_v1)