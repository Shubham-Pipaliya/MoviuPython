from mongoengine import (
    Document, StringField, FloatField, BooleanField,
    ListField, DateTimeField, IntField, ObjectIdField
)

class Movie(Document):
    meta = {'collection': 'movies', 'db_alias': 'default', 'strict': False}
    title           = StringField(required=True)
    description     = StringField()
    release_date    = DateTimeField()
    genre           = StringField()
    director        = StringField()
    writer          = StringField()
    cast            = ListField(StringField())
    banner_url      = StringField()
    poster_url      = StringField()
    rating          = FloatField(default=0.0)
    language        = StringField()
    runtime         = IntField()
    is_deleted      = BooleanField(default=False)
    deleted_at      = DateTimeField()
    created_at      = DateTimeField()
    updated_at      = DateTimeField()
    __v             = IntField()
    trailer_url     = StringField()
    now_streaming_on   = ListField(StringField())


class MovieReview(Document):
    meta = {
        'collection': 'reviews',  # ✅ corrected from 'moviereviews'
        'db_alias': 'default'
    }
    user = StringField(required=True)
    movie = StringField(required=True)
    rating = FloatField(required=True)
    is_deleted = BooleanField(default=False)


class TVShow(Document):
    meta = {
        'collection': 'shows',
        'db_alias': 'default',
        'strict': False
    }
    title           = StringField(required=True)
    description     = StringField()
    release_date    = DateTimeField()
    genre           = StringField()
    director        = StringField()
    writer          = StringField()
    cast            = ListField(StringField())
    banner_url      = StringField()
    poster_url      = StringField()
    rating          = FloatField(default=0.0)
    language        = StringField()
    runtime         = IntField()
    is_deleted      = BooleanField(default=False)
    deleted_at      = DateTimeField()
    created_at      = DateTimeField()
    updated_at      = DateTimeField()
    __v             = IntField()
    trailer_url     = StringField()
    season_year     = IntField()
    season          = IntField()
    trailer_url     = StringField()
    now_streaming_on   = ListField(StringField())


class ReviewShow(Document):
    meta = {
        'collection': 'reviewshows',  # ✅ already correct
        'db_alias': 'default'
    }
    user = StringField(required=True)
    show = StringField(required=True)
    rating = FloatField(required=True)
    is_deleted = BooleanField(default=False)

class Following(Document):
    meta = {
        'collection': 'followings',
        'db_alias': 'default'
    }
    userId = StringField(required=True)
    followingId = StringField(required=True)
class User(Document):
    username = StringField()
    email = StringField()
    genres = ListField(ObjectIdField())  # ✅ fix

    meta = {
        "collection": "users",
        "strict": False
    }


class Genre(Document):
    name = StringField(required=True)
    meta = {
        "collection": "genres",
        "strict": False
    }