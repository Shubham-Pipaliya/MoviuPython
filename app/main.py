from app import db 
from fastapi import FastAPI, HTTPException
from app.recommendation_model import load_model, get_top_n, predict_rating
from app.tv_show_recommender import load_show_model, get_top_n_shows, predict_show_rating
from app.models import Movie, TVShow
from app.utils.mongo_data_loader import get_movies_df
from bson import ObjectId
from app.utils.mongo_data_loader import get_shows_df


app = FastAPI(title="Recommendation API")

# --- Load models at startup ---
try:
    movie_model, movie_data, movie_reviews_df = load_model()
    show_model, show_data, show_reviews_df = load_show_model()
except Exception as e:
    print("Model loading failed:", e)
    raise


# --- Health Check ---
@app.get("/")
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
