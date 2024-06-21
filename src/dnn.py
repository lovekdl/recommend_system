import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
import os

# 自定义数据集
class MovieRatingDataset(Dataset):
    def __init__(self, data, user_to_index, movie_to_index, unknown_user_id, unknown_movie_id):
        self.users = torch.tensor([user_to_index.get(user_id, unknown_user_id) for user_id in data['userId'].values], dtype=torch.long)
        self.movies = torch.tensor([movie_to_index.get(movie_id, unknown_movie_id) for movie_id in data['movieId'].values], dtype=torch.long)
        self.ratings = torch.tensor(data['rating'].values, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

# 神经网络模型
class DNNModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=128, hidden_dim=256):
        super(DNNModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)  # 加1处理未知用户
        self.movie_embedding = nn.Embedding(num_movies + 1, embedding_dim)  # 加1处理未知电影
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        x = torch.cat([user_embeds, movie_embeds], dim=1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x.squeeze()

def train_model(model, train_loader, test_loader, device, num_epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    # print(num_epochs)
    # print(type(num_epochs))
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for user_ids, movie_ids, ratings in train_loader:
            user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)
            optimizer.zero_grad()
            outputs = model(user_ids, movie_ids)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * user_ids.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for user_ids, movie_ids, ratings in test_loader:
                user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)
                outputs = model(user_ids, movie_ids)
                loss = criterion(outputs, ratings)
                test_loss += loss.item() * user_ids.size(0)

        test_loss /= len(test_loader.dataset)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    # Save the model
    # torch.save(model.state_dict(), 'dnn_recommender.pth')

def predict_and_evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    true_ratings = []
    with torch.no_grad():
        for user_ids, movie_ids, ratings in data_loader:
            user_ids, movie_ids = user_ids.to(device), movie_ids.to(device)
            outputs = model(user_ids, movie_ids)
            predictions.extend(outputs.cpu().numpy())
            true_ratings.extend(ratings.numpy())
    return predictions

def dnn(train, test, epoch=10, batch_size=64, save_dir="data/icf"):
    user_ids = train['userId'].unique()
    movie_ids = train['movieId'].unique()

    num_users = len(user_ids)
    num_movies = len(movie_ids)

    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    unknown_user_id = num_users
    unknown_movie_id = num_movies

    # 创建数据加载器
    train_dataset = MovieRatingDataset(train, user_to_index, movie_to_index, unknown_user_id, unknown_movie_id)
    test_dataset = MovieRatingDataset(test, user_to_index, movie_to_index, unknown_user_id, unknown_movie_id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DNNModel(num_users, num_movies)
    train_model(model, train_loader, test_loader, device, num_epochs=epoch, lr=1e-4)

    # Predict and evaluate on train and test sets
    train_predictions = predict_and_evaluate(model, train_loader, device)
    test_predictions = predict_and_evaluate(model, test_loader, device)

    

    # Save predictions
    train['score'] = train_predictions
    test['score'] = test_predictions
    train_rmse = np.sqrt(((train['rating'] - train['score']) ** 2).mean())
    test_rmse = np.sqrt(((test['rating'] - test['score']) ** 2).mean())
    print(f"Train RMSE: {train_rmse}")
    print(f"Test RMSE: {test_rmse}")
    os.makedirs(save_dir, exist_ok=True)
    train.to_csv(f"{save_dir}/train_predictions.csv", index=False)
    test.to_csv(f"{save_dir}/test_predictions.csv", index=False)
    return train_rmse, test_rmse