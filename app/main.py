from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app import db
from app.recommendation_model import load_model, get_top_n, predict_rating
from app.tv_show_recommender import load_show_model, get_top_n_shows, predict_show_rating
from app.models import Movie, TVShow
from app.utils.mongo_data_loader import get_movies_df, get_shows_df
from bson import ObjectId
import redis
import time

app = FastAPI(title="Recommendation API")

# --- Mount static directory ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_home():
    return FileResponse("static/index.html")

# --- Load models at startup ---
try:
    movie_model, movie_data, movie_reviews_df = load_model()
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
        trending_ids = trending['movie_id'].head(n).tolist()
        return {
            "language": language,
            "user_id": user_id,
            "recommendations": trending_ids
        }

    movie_ids = [mid for mid, _ in recs]
    return {
        "language": language,
        "user_id": user_id,
        "recommendations": movie_ids
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
