from mongoengine import Document, StringField, FloatField, BooleanField, ListField, ObjectIdField

class Movie(Document):
    meta = {
        'collection': 'movies',
        'db_alias': 'default'  # ✅ REQUIRED
    }
    title = StringField(required=True)
    genre = StringField()
    language = StringField()
    release_date = StringField()  # ✅ added for consistency
    rating = FloatField(default=0.0)  # ✅ added for consistency


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
        'db_alias': 'default'
    }
    title = StringField(required=True)
    genre = StringField()
    language = StringField()
    release_date = StringField()  # ✅ added for consistency
    rating = FloatField(default=0.0)  # ✅ added for consistency


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