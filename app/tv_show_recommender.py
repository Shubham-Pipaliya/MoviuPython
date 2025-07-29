import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

from app.utils.mongo_data_loader import (
    get_review_shows_df,
    get_shows_df,
    get_followings_df
)

# Globals
show_genre_matrix = None
show_genre_vectors = None
followings_map = {}


def load_show_model():
    global show_genre_matrix, show_genre_vectors, followings_map

    df = get_review_shows_df()
    shows_meta = get_shows_df()
    followings_df = get_followings_df()

    if df.empty or "rating" not in df.columns:
        raise ValueError("❌ No valid review show data with 'user', 'show', 'rating'")

    # --- Followings ---
    followings_df = followings_df.dropna(subset=["user", "following"])
    followings_map = followings_df.groupby("user")["following"].apply(list).to_dict()

    # --- Genre Matrix ---
    shows_meta["genre_list"] = shows_meta["genre"].fillna("").apply(
        lambda x: [g.strip() for g in x.split(",") if g.strip()]
    )
    mlb = MultiLabelBinarizer()
    show_genre_matrix = pd.DataFrame(
        mlb.fit_transform(shows_meta["genre_list"]),
        columns=mlb.classes_,
        index=shows_meta["show_id"]
    )
    show_genre_vectors = show_genre_matrix.copy()

    # --- Collaborative Filtering ---
    reader = Reader(rating_scale=(df["rating"].min(), df["rating"].max()))
    data = Dataset.load_from_df(df[["user", "show", "rating"]], reader)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
    algo = SVD()
    algo.fit(trainset)

    print("✅ Show model trained successfully")
    return algo, data, df


def get_top_n_shows(algo, data, df_reviews, n=10, hybrid=True, language_filter=None, metadata_df=None):
    global show_genre_vectors, followings_map
    trainset = data.build_full_trainset()
    all_items = set(trainset._raw2inner_id_items.keys())
    all_users = set(trainset._raw2inner_id_users.keys())

    if language_filter and metadata_df is not None:
        filtered_ids = set(
            metadata_df[metadata_df["language"].str.lower() == language_filter.lower()]["show_id"]
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
                if not user_rated.empty:
                    rated_vectors = show_genre_vectors.loc[
                        show_genre_vectors.index.intersection(user_rated["show"])
                    ].values
                    user_vector = np.mean(rated_vectors, axis=0).reshape(1, -1)
                    target_vector = (
                        show_genre_vectors.loc[iid].values.reshape(1, -1)
                        if iid in show_genre_vectors.index
                        else np.zeros((1, len(user_vector[0])))
                    )
                    genre_score = cosine_similarity(user_vector, target_vector)[0][0]

                followings = followings_map.get(uid, [])
                follower_ratings = df_reviews[
                    (df_reviews["user"].isin(followings)) & (df_reviews["show"] == iid)
                ]
                if not follower_ratings.empty:
                    follower_score = follower_ratings["rating"].mean()

                final_score = 0.6 * cf_score + 0.25 * genre_score + 0.15 * follower_score
            else:
                final_score = cf_score

            predictions.append((iid, final_score))

        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [iid for iid, _ in predictions[:n]]

    return top_n

def get_fallback_shows_for_user(user_id, language, show_reviews_df, followings_df, show_df, n=10):
    global show_genre_vectors

    # Followings-based fallback
    followings = followings_df[followings_df["user"] == user_id]["following"].tolist()
    if followings:
        peer_reviews = show_reviews_df[show_reviews_df["user"].isin(followings)]
        peer_shows = peer_reviews.merge(show_df, left_on="show", right_on="show_id")
        peer_shows = peer_shows[peer_shows["language"].str.lower() == language.lower()]
        if not peer_shows.empty:
            top_peer_shows = (
                peer_shows.groupby(["show_id", "title", "genre", "language", "poster_url"])
                .agg(avg_rating=("rating", "mean"))
                .reset_index()
                .sort_values("avg_rating", ascending=False)
                .head(n)
            )
            return top_peer_shows.to_dict(orient="records")

    # Genre-based fallback (real user vector or default)
    user_rated = show_reviews_df[(show_reviews_df["user"] == user_id) & (show_reviews_df["rating"] >= 4.0)]

    if not user_rated.empty:
        rated_vectors = show_genre_vectors.loc[
            show_genre_vectors.index.intersection(user_rated["show"])
        ].values
        user_vector = np.nan_to_num(np.mean(rated_vectors, axis=0)).reshape(1, -1)
    else:
        # Default genre vector
        default_genres = ["Action", "Thriller", "Drama"]
        default_vector = np.zeros((1, show_genre_vectors.shape[1]))
        for genre in default_genres:
            if genre in show_genre_vectors.columns:
                default_vector[0][show_genre_vectors.columns.get_loc(genre)] = 1.0
        user_vector = default_vector

    # Candidate shows by language
    candidate_shows = show_df[
        show_df["language"].str.lower() == language.lower()
    ].copy()

    candidate_ids = candidate_shows["show_id"].tolist()
    candidate_vectors = show_genre_vectors.loc[
        show_genre_vectors.index.intersection(candidate_ids)
    ]

    if candidate_vectors.empty:
        return None

    # Compute genre similarity
    similarities = cosine_similarity(user_vector, candidate_vectors.fillna(0).values)[0]
    candidate_shows["similarity"] = similarities
    fallback = candidate_shows.sort_values("similarity", ascending=False).head(n)

    return fallback[["show_id", "title", "genre", "language", "poster_url"]].to_dict(orient="records")


def predict_show_rating(algo, user_id, show_id):
    pred = algo.predict(uid=str(user_id), iid=str(show_id))
    return pred.est

def get_show_genre_vectors():
    global show_genre_vectors
    return show_genre_vectors

