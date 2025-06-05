import pandas as pd
from app import db  # ensure connection is triggered
from app.models import (
    Movie,
    TVShow,
    MovieReview,  # ✅ fix here
    ReviewShow,
    Following
)

# --- Load movie reviews ---
def get_reviews_df():
    review_qs = MovieReview.objects(is_deleted=False).only("user", "movie", "rating")
    return pd.DataFrame([{
        "user": r.user,
        "movie": r.movie,
        "rating": r.rating
    } for r in review_qs])

# --- Load TV show reviews ---
def get_review_shows_df():
    reviews = ReviewShow.objects(is_deleted=False).only("user", "show", "rating")
    return pd.DataFrame([{
        "user": r.user,
        "show": r.show,
        "rating": r.rating
    } for r in reviews])

# --- Load movie metadata ---
def get_movies_df():
    movies = Movie.objects.only("id", "title", "genre", "language")
    return pd.DataFrame([{
        "movie_id": str(m.id),
        "title": m.title,
        "genre": m.genre or "",
        "language": m.language or ""
    } for m in movies])

# --- Load show metadata ---
def get_shows_df():
    shows = TVShow.objects.only("id", "title", "genre", "language")
    return pd.DataFrame([{
        "show_id": str(s.id),
        "title": s.title,
        "genre": s.genre or "",
        "language": s.language or ""
    } for s in shows])

# --- Load followings ---
def get_followings_df():
    followings = Following.objects.only("userId", "followingId")
    return pd.DataFrame([{
        "user": f.userId,
        "following": f.followingId
    } for f in followings])
