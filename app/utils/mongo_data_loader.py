import pandas as pd
import re
from app import db  # ensure connection is triggered
from app.models import (
    Movie,
    TVShow,
    MovieReview,  # ✅ fix here
    ReviewShow,
    Following,
    User
)

# --- Load movie reviews ---
def get_reviews_df():
    review_qs = MovieReview.objects(is_deleted=False).only("user", "movie", "rating")
    return pd.DataFrame([{
        "user": r.user,
        "movie": r.movie,
        "rating": r.rating
    } for r in review_qs])

def slugify(text):
    text = re.sub(r'[^a-zA-Z0-9]+', '-', text.lower()).strip('-')
    return text

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
    movies = Movie.objects  # load full docs
    return pd.DataFrame([{
        "movie_id":        str(m.id),
        "title":           m.title,
        "description":     m.description or "",
        "release_date":    m.release_date.isoformat() if m.release_date else "",
        "rating":          m.rating or 0.0,
        "genre":           m.genre or "",
        "language":        m.language or "",
        "banner_url":      m.banner_url,
        "poster_url":      m.poster_url,
        "cast":            m.cast or [],
        "director":        m.director or "",
        "writer":          m.writer or "",
        "runtime":         m.runtime or 0,
        "is_deleted":      m.is_deleted,
        "deleted_at":      m.deleted_at.isoformat() if m.deleted_at else None,
        "created_at":      m.created_at.isoformat() if m.created_at else None,
        "updated_at":      m.updated_at.isoformat() if m.updated_at else None,
        "__v":             m._data.get("__v", 0),
    } for m in movies])


# --- Load show metadata ---
def get_shows_df():
    shows = TVShow.objects.only("id", "title", "genre", "language")
    return pd.DataFrame([{
        "show_id":        str(m.id),
        "title":           m.title,
        "description":     m.description or "",
        "release_date":    m.release_date.isoformat() if m.release_date else "",
        "rating":          m.rating or 0.0,
        "genre":           m.genre or "",
        "language":        m.language or "",
        "banner_url":      m.banner_url,
        "poster_url":      m.poster_url,
        "cast":            m.cast or [],
        "director":        m.director or "",
        "writer":          m.writer or "",
        "runtime":         m.runtime or 0,
        "is_deleted":      m.is_deleted,
        "deleted_at":      m.deleted_at.isoformat() if m.deleted_at else None,
        "created_at":      m.created_at.isoformat() if m.created_at else None,
        "updated_at":      m.updated_at.isoformat() if m.updated_at else None,
        "__v":             m._data.get("__v", 0),
    } for m in shows])

# --- Load followings ---
def get_followings_df():
    followings = Following.objects.only("userId", "followingId")
    return pd.DataFrame([{
        "user": f.userId,
        "following": f.followingId
    } for f in followings])

# --- Load User data ---

def get_user_genres(user_id: str):
    user = User.objects(id=user_id).first()
    return user.genres if user and user.genres else []
