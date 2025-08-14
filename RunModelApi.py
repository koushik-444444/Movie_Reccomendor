from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Any

# Initialize FastAPI application
app = FastAPI(title="Movie Recommendation API", description="An API to recommend movies based on user preferences.", version="1.0")

# Load data and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '/content/implementation/model/best_model.pt'

# Placeholder variables (replace with actual values from your training setup)
n_users = 610  # Number of users
n_items = 9724  # Number of movies

# Load your dataset (ensure these paths and data are correct)
movies_df = pd.read_csv('data/ml-latest-small/movies.csv')
ratings_df = pd.read_csv('data/ml-latest-small/ratings.csv')

# Load mappings from the training set
train_set = ...  # Your MovieDataset object
movie_names = movies_df.set_index('movieId')['title'].to_dict()

# Initialize the model
model = ImprovedRecommender(n_users=n_users, n_items=n_items, n_factors=50, n_genres=train_set.n_genre_features)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

# Request and Response models
class RecommendationRequest(BaseModel):
    user_id: int
    movie_id: int
    n_recommendations: int = 10

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]

# Recommendation logic
def recommend_movies(user_id, movie_id, model, movies_df, train_set, n_recommendations=10):
    if user_id not in train_set.userid2idx or movie_id not in train_set.movieid2idx:
        raise HTTPException(status_code=404, detail="User ID or Movie ID not found in training data.")

    user_idx = train_set.userid2idx[user_id]
    movie_idx = train_set.movieid2idx[movie_id]

    # Prepare temporal and genre features
    movie_features = movies_df.loc[movies_df['movieId'] == movie_id].iloc[0]
    genres_list = movie_features['genres'].split('|')
    genre_indices = [train_set.ratings.columns.get_loc(genre) - 4 for genre in genres_list if genre in train_set.ratings.columns]
    genres = np.zeros(train_set.n_genre_features)
    genres[genre_indices] = 1

    scaler = MinMaxScaler()
    temporal_features = scaler.fit_transform(np.array([[12, 2]]))

    feature_vector = np.concatenate((
        np.array([user_idx, movie_idx]),
        temporal_features[0],
        genres
    ))

    feature_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(feature_tensor).cpu().numpy()

    scores = []
    for candidate_movie_idx in range(len(train_set.movieid2idx)):
        if candidate_movie_idx == movie_idx:
            continue

        candidate_feature_vector = np.concatenate((
            np.array([user_idx, candidate_movie_idx]),
            temporal_features[0],
            genres
        ))

        candidate_tensor = torch.FloatTensor(candidate_feature_vector).unsqueeze(0).to(device)
        with torch.no_grad():
            score = model(candidate_tensor).cpu().numpy().flatten()[0]
            scores.append((candidate_movie_idx, score))

    top_recommendations = sorted(scores, key=lambda x: x[1], reverse=True)[:n_recommendations]

    recommendations = [
        {"movie_id": train_set.idx2movieid[movie_idx], "title": movie_names[train_set.idx2movieid[movie_idx]]}
        for movie_idx, _ in top_recommendations
    ]

    return recommendations

# Define API endpoint
@app.post("/recommendations", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    try:
        recommendations = recommend_movies(
            user_id=request.user_id,
            movie_id=request.movie_id,
            model=model,
            movies_df=movies_df,
            train_set=train_set,
            n_recommendations=request.n_recommendations
        )
        return RecommendationResponse(recommendations=recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
# To run the app, use: uvicorn RunModelApi:app --reload
