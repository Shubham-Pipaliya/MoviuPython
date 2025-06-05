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
    mlb = MultiLabelBinarizer()
    genre_matrix_df = pd.DataFrame(
        mlb.fit_transform(movies_meta["genre_list"]),
        columns=mlb.classes_,
        index=movies_meta["movie_id"]
    )
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

def predict_rating(algo, user_id, movie_id):
    pred = algo.predict(uid=str(user_id), iid=str(movie_id))
    return pred.est
