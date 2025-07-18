from mongoengine import Document, StringField, FloatField, BooleanField, ListField, ObjectIdField

class Movie(Document):
    meta = {
        'collection': 'movies',
        'db_alias': 'default'  # ✅ REQUIRED
    }
    title = StringField(required=True)
    genre = StringField()
    language = StringField()


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
