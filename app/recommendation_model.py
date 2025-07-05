import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

from app.utils.mongo_data_loader import (
    get_reviews_df,
    get_movies_df,
    get_followings_df
)

# Globals
genre_matrix = None
movie_genre_vectors = None
followings_map = {}

def load_model():
    global genre_matrix, movie_genre_vectors, followings_map

    # --- Load data from MongoDB via ORM ---
    df = get_reviews_df().dropna(subset=["user", "movie", "rating"])
    movies_meta = get_movies_df().fillna({"genre": "", "language": ""})
    print("[DEBUG] Movies loaded:", movies_meta.shape)
    print("[DEBUG] Sample genres:", movies_meta["genre"].head())

    followings_df = get_followings_df().dropna(subset=["user", "following"])

    if df.empty:
        raise ValueError("❌ No review data found.")
    if "rating" not in df.columns:
        raise ValueError("❌ 'rating' column missing in review data.")

    # --- Followings Map ---
    if not {"user", "following"}.issubset(followings_df.columns):
        raise ValueError("❌ Followings data missing required columns.")

    followings_map = followings_df.groupby("user")["following"].apply(list).to_dict()

    # --- Genre Matrix ---
    movies_meta["genre_list"] = movies_meta["genre"].apply(
        lambda x: [g.strip() for g in x.split(",") if g.strip()]
    )
    print("[DEBUG] Sample genre_list:", movies_meta["genre_list"].head())
    mlb = MultiLabelBinarizer()
    genre_matrix_df = pd.DataFrame(
        mlb.fit_transform(movies_meta["genre_list"]),
        columns=mlb.classes_,
        index=movies_meta["movie_id"]
    )
    print("[DEBUG] genre_matrix_df shape:", genre_matrix_df.shape)
    print("[DEBUG] genre_matrix_df columns:", list(genre_matrix_df.columns))

    movie_genre_vectors = genre_matrix_df.fillna(0).copy()

    # --- Collaborative Filtering ---
    reader = Reader(rating_scale=(df["rating"].min(), df["rating"].max()))
    data = Dataset.load_from_df(df[["user", "movie", "rating"]], reader)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
    algo = SVD()
    algo.fit(trainset)

    print("✅ Movie model trained successfully")
    return algo, data, df

def get_top_n(algo, data, df_reviews, n=10, hybrid=True, language_filter=None, metadata_df=None):
    global movie_genre_vectors, followings_map
    trainset = data.build_full_trainset()
    all_items = set(trainset._raw2inner_id_items.keys())
    all_users = set(trainset._raw2inner_id_users.keys())

    if language_filter and metadata_df is not None:
        filtered_ids = set(
            metadata_df[metadata_df["language"].str.lower() == language_filter.lower()]["movie_id"]
        )
        all_items = all_items.intersection(filtered_ids)

    top_n = defaultdict(list)

    for uid in all_users:
        try:
            inner_uid = trainset.to_inner_uid(uid)
            seen_items = set(trainset.to_raw_iid(inner_iid) for inner_iid, _ in trainset.ur[inner_uid])
        except ValueError:
            continue

        unseen_items = all_items - seen_items
        predictions = []

        for iid in unseen_items:
            cf_score = algo.predict(uid, iid).est
            genre_score = 0
            follower_score = 0

            if hybrid:
                user_rated = df_reviews[(df_reviews["user"] == uid) & (df_reviews["rating"] >= 4.0)]
                rated_vectors = movie_genre_vectors.loc[
                    movie_genre_vectors.index.intersection(user_rated["movie"])
                ].values if not user_rated.empty else np.array([])

                if rated_vectors.size == 0:
                    user_vector = np.zeros((1, movie_genre_vectors.shape[1]))
                else:
                    user_vector = np.nan_to_num(np.mean(rated_vectors, axis=0)).reshape(1, -1)

                target_vector = (
                    movie_genre_vectors.loc[iid].values.reshape(1, -1)
                    if iid in movie_genre_vectors.index
                    else np.zeros((1, movie_genre_vectors.shape[1]))
                )
                genre_score = cosine_similarity(user_vector, target_vector)[0][0]

                followings = followings_map.get(uid, [])
                follower_ratings = df_reviews[
                    (df_reviews["user"].isin(followings)) & (df_reviews["movie"] == iid)
                ]
                if not follower_ratings.empty:
                    follower_score = follower_ratings["rating"].mean()

                final_score = 0.6 * cf_score + 0.25 * genre_score + 0.15 * follower_score
            else:
                final_score = cf_score

            predictions.append((iid, final_score))

        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = predictions[:n]

    return top_n

def get_fallback_movies_for_user(user_id, language, movie_reviews_df, followings_df, movies_df, n=10):
    global movie_genre_vectors

    # Followings-based fallback
    followings = followings_df[followings_df["user"] == user_id]["following"].tolist()
    if followings:
        peer_reviews = movie_reviews_df[movie_reviews_df["user"].isin(followings)]
        peer_movies = peer_reviews.merge(movies_df, left_on="movie", right_on="movie_id")
        peer_movies = peer_movies[peer_movies["language"].str.lower() == language.lower()]
        if not peer_movies.empty:
            top_peer_movies = (
                peer_movies.groupby(["movie_id", "title", "genre", "language", "poster_url"])
                .agg(avg_rating=("rating", "mean"))
                .reset_index()
                .sort_values("avg_rating", ascending=False)
                .head(n)
            )
            return top_peer_movies.to_dict(orient="records")

    # Genre-based fallback (real user vector or default)
    user_rated = movie_reviews_df[(movie_reviews_df["user"] == user_id) & (movie_reviews_df["rating"] >= 4.0)]

    if not user_rated.empty:
        rated_vectors = movie_genre_vectors.loc[
            movie_genre_vectors.index.intersection(user_rated["movie"])
        ].values
        user_vector = np.nan_to_num(np.mean(rated_vectors, axis=0)).reshape(1, -1)
    else:
        # Default genre vector
        default_genres = ["Action", "Thriller", "Drama"]
        default_vector = np.zeros((1, movie_genre_vectors.shape[1]))
        for genre in default_genres:
            if genre in movie_genre_vectors.columns:
                default_vector[0][movie_genre_vectors.columns.get_loc(genre)] = 1.0
        user_vector = default_vector

    # Candidate movies by language
    candidate_movies = movies_df[
        movies_df["language"].str.lower() == language.lower()
    ].copy()

    candidate_ids = candidate_movies["movie_id"].tolist()
    candidate_vectors = movie_genre_vectors.loc[
        movie_genre_vectors.index.intersection(candidate_ids)
    ]

    if candidate_vectors.empty:
        return None

    # Compute genre similarity
    similarities = cosine_similarity(user_vector, candidate_vectors.fillna(0).values)[0]
    candidate_movies["similarity"] = similarities
    fallback = candidate_movies.sort_values("similarity", ascending=False).head(n)

    return fallback[["movie_id", "title", "genre", "language", "poster_url"]].to_dict(orient="records")

def predict_rating(algo, user_id, movie_id):
    pred = algo.predict(uid=str(user_id), iid=str(movie_id))
    return pred.est

def get_movie_genre_vectors():
    global movie_genre_vectors
    return movie_genre_vectors
