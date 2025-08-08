from bson import ObjectId
from app.models import Movie, MovieReview, TVShow, ReviewShow

def _compute_stats(review_cls, id_field, ids):
    # returns a map { str(id) → {"avgRating":…, "reviewCount":…} }
    pipeline = [
        {"$match": {id_field: {"$in": ids}, "is_deleted": False}},
        {"$group": {
            "_id": f"${id_field}",
            "avgRating": {"$avg": "$rating"},
            "reviewCount": {"$sum": 1}
        }}
    ]
    stats = { str(s["_id"]): s for s in review_cls.objects.aggregate(*pipeline) }
    return stats

def serialize_movies(movie_ids):
    stats_map = _compute_stats(MovieReview, "movie", [ObjectId(m) for m in movie_ids])
    docs = Movie.objects(id__in=[ObjectId(m) for m in movie_ids])
    out = []
    for doc in docs:
        sid = str(doc.id)
        st = stats_map.get(sid, {"avgRating": 0, "reviewCount": 0})
        out.append({
            "_id": sid,
            "__v": doc._data.get("__v", 0),
            "avgRating": round(st["avgRating"], 2),
            "reviewCount": st["reviewCount"],
            "title": doc.title,
            "banner_url": doc.banner_url,
            "cast": doc.cast or [],
            "description": doc.description,
            "director": doc.director,
            "writer": doc.writer,
            "genre": doc.genre,
            "language": doc.language,
            "poster_url": doc.poster_url,
            "rating": doc.rating,
            "release_date": doc.release_date.isoformat() if doc.release_date else None,
            "runtime": doc.runtime,
            "is_deleted": doc.is_deleted,
            "trailer_url": doc.trailer_url,
            "now_streaming_on": doc.now_streaming_on or [],
            "deleted_at": doc.deleted_at.isoformat() if doc.deleted_at else None,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
        })
    return out

def serialize_shows(show_ids):
    stats_map = _compute_stats(ReviewShow, "show", [ObjectId(s) for s in show_ids])
    docs = TVShow.objects(id__in=[ObjectId(s) for s in show_ids])
    out = []
    for doc in docs:
        sid = str(doc.id)
        st = stats_map.get(sid, {"avgRating": 0, "reviewCount": 0})
        out.append({
            "_id": sid,
            "__v": doc._data.get("__v", 0),
            "avgRating": round(st["avgRating"], 2),
            "reviewCount": st["reviewCount"],
            "title": doc.title,
            "banner_url": doc.banner_url,
            "cast": doc.cast or [],
            "description": doc.description,
            "director": doc.director,
            "writer": doc.writer,
            "genre": doc.genre,
            "language": doc.language,
            "poster_url": doc.poster_url,
            "rating": doc.rating,
            "release_date": doc.release_date.isoformat() if doc.release_date else None,
            "runtime": doc.runtime,
            "trailer_url": doc.trailer_url,
            "now_streaming_on": doc.now_streaming_on or [],
            "is_deleted": doc.is_deleted,
            "deleted_at": doc.deleted_at.isoformat() if doc.deleted_at else None,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
        })
    return out
        
