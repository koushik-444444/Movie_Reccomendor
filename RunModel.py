import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '/content/implementation/model/best_model.pt'

# Initialize the model (make sure to use the same architecture and parameters)
model = ImprovedRecommender(n_users=n_users, n_items=n_items, n_factors=50, n_genres=train_set.n_genre_features) # Changed Recommender to ImprovedRecommender
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

# Function to recommend movies
def recommend_movies(user_id, movie_id, model, movies_df, train_set, n_recommendations=10):
    """
    Recommend movies for a given user and movie.

    Args:
    - user_id: int, the user ID
    - movie_id: int, the movie ID
    - model: PyTorch model
    - movies_df: DataFrame, movie metadata
    - train_set: MovieDataset, to map indices
    - n_recommendations: int, number of recommendations

    Returns:
    - List of recommended movies
    """
    # Convert IDs to indices using mappings
    if user_id not in train_set.userid2idx or movie_id not in train_set.movieid2idx:
        return "User ID or Movie ID not in the training data."

    user_idx = train_set.userid2idx[user_id]
    movie_idx = train_set.movieid2idx[movie_id]

    # Prepare temporal and genre features for this movie
    movie_features = movies_df.loc[movies_df['movieId'] == movie_id].iloc[0]
    genres = movie_features.iloc[2:].values  # Extract genres
    # genres = genres.astype(float)
    genres_list = movie_features['genres'].split('|')  # Split the genre string
    genre_indices = [train_set.ratings.columns.get_loc(genre) - 4 for genre in genres_list if genre in train_set.ratings.columns]  # Get indices of genres
    genres = np.zeros(train_set.n_genre_features)  # Create a zero-filled array
    genres[genre_indices] = 1  # Set the corresponding genre indices to 1

    # Simulate temporal features (scaled to match training)
    scaler = MinMaxScaler()
    temporal_features = scaler.fit_transform(np.array([[12, 2]]))  # Example hour=12, day_of_week=2 (Tuesday)
    
    # Create feature vector for the user and input movie
    feature_vector = np.concatenate((
        np.array([user_idx, movie_idx]),  # User and movie indices
        temporal_features[0],  # Scaled temporal features
        genres  # Genre features
    ))

    # Convert to tensor and pass through the model
    feature_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(feature_tensor).cpu().numpy()

    # Rank all movies for the user
    scores = []
    for candidate_movie_idx in range(len(train_set.movieid2idx)):
        if candidate_movie_idx == movie_idx:  # Skip the current movie
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

    # Sort movies by predicted scores in descending order
    top_recommendations = sorted(scores, key=lambda x: x[1], reverse=True)[:n_recommendations]

    # Get movie names from indices
    recommendations = [(train_set.idx2movieid[movie_idx], movie_names[train_set.idx2movieid[movie_idx]])
                       for movie_idx, _ in top_recommendations]
    return recommendations

# Example usage
user_id = 1
movie_id = 1  # Replace with the actual movieId
recommended_movies = recommend_movies(user_id, movie_id, model, movies_df, train_set, n_recommendations=10)

print("Recommended Movies:")
for movie in recommended_movies:
    print(f"Movie ID: {movie[0]}, Title: {movie[1]}")
