import logging
logger = logging.getLogger(__name__)

@app.get("/predict/")
def predict(user_id: str, movie_id: str):
    logger.info(f"Predicting for user: {user_id}, movie: {movie_id}")
    ...
