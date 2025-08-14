# Movie Recommendation System
# This script implements a movie recommendation system using PyTorch.

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import zipfile
import os
from pathlib import Path

# Ensure necessary directories exist
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# Download the dataset
if not os.path.exists('ml-latest-small.zip'):
    os.system('curl http://files.grouplens.org/datasets/movielens/ml-latest-small.zip -o ml-latest-small.zip')

with zipfile.ZipFile('ml-latest-small.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

# Import the dataset
movies_df = pd.read_csv('data/ml-latest-small/movies.csv')
ratings_df = pd.read_csv('data/ml-latest-small/ratings.csv')

# Dataset Class
class MovieDataset(Dataset):
    def __init__(self, ratings_df, movies_df, train=True):
        self.ratings = ratings_df.copy()
        self.ratings['datetime'] = pd.to_datetime(self.ratings['timestamp'], unit='s')
        self.ratings['hour'] = self.ratings['datetime'].dt.hour
        self.ratings['day_of_week'] = self.ratings['datetime'].dt.dayofweek

        # Add movie features
        movies = movies_df.copy()
        genre_dummies = movies['genres'].str.get_dummies('|')
        self.n_genre_features = len(genre_dummies.columns)
        movies = pd.concat([movies, genre_dummies], axis=1)
        self.ratings = self.ratings.merge(movies[['movieId'] + list(genre_dummies.columns)], on='movieId', how='left')

        # Create continuous IDs
        self.userid2idx = {o: i for i, o in enumerate(self.ratings['userId'].unique())}
        self.movieid2idx = {o: i for i, o in enumerate(self.ratings['movieId'].unique())}
        self.idx2movieid = {i: o for o, i in self.movieid2idx.items()}

        self.ratings['user_idx'] = self.ratings['userId'].map(self.userid2idx)
        self.ratings['movie_idx'] = self.ratings['movieId'].map(self.movieid2idx)

        if train:
            self.ratings = self.ratings.sample(frac=0.8, random_state=42)
        else:
            self.ratings = self.ratings.sample(frac=0.2, random_state=42)

        scaler = MinMaxScaler()
        self.ratings[['hour', 'day_of_week']] = scaler.fit_transform(self.ratings[['hour', 'day_of_week']])

        self.features = self.ratings[['user_idx', 'movie_idx', 'hour', 'day_of_week'] + list(genre_dummies.columns)].values
        self.targets = self.ratings['rating'].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.targets[idx]])

# Recommender Model
class Recommender(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=50, n_genres=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(n_factors * 2 + 2 + n_genres, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, features):
        user_idx = features[:, 0].long()
        movie_idx = features[:, 1].long()
        temporal_features = features[:, 2:4]
        genre_features = features[:, 4:]

        user_embedding = self.user_factors(user_idx)
        movie_embedding = self.item_factors(movie_idx)

        x = torch.cat([user_embedding, movie_embedding, temporal_features, genre_features], dim=1)
        return self.nn(x)

# Prepare dataset
n_users = len(ratings_df.userId.unique())
n_items = len(ratings_df.movieId.unique())
train_set = MovieDataset(ratings_df, movies_df, train=True)
val_set = MovieDataset(ratings_df, movies_df, train=False)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)

# Training the Model
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    ensure_dir('implementation/model')  # Ensure the model directory exists
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = torch.nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, targets).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'implementation/model/best_model.pt')

        print(f'Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')

# Initialize and train model
model = Recommender(n_users, n_items, n_factors=50, n_genres=20)
train_model(model, train_loader, val_loader, epochs=128, lr=1e-3)
