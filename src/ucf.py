import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

class ucf_model:
    def __init__(self, train, top_k=50):
        self.train = train
        self.top_k = top_k
        self.user_item_matrix = train.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        self.similar_users = self.precompute_similar_users()
        
    def jaccard_similarity(self, user1, user2):
        user1_movies = set(self.train[(self.train['userId'] == user1) & (self.train['rating'] >= 4)]['movieId'])
        user2_movies = set(self.train[(self.train['userId'] == user2) & (self.train['rating'] >= 4)]['movieId'])
        
        intersection = len(user1_movies.intersection(user2_movies))
        union = len(user1_movies.union(user2_movies))
        
        if union == 0:
            return 0
        
        return intersection / union

    def precompute_similar_users(self):
        user_ids = self.user_item_matrix.index
        similar_users = Parallel(n_jobs=-1)(delayed(self.get_top_n_similar_users)(user_id, n=self.top_k) for user_id in tqdm(user_ids, desc="Processing Similar Users"))
        return dict(zip(user_ids, similar_users))

    def get_top_n_similar_users(self, user_id, n=50):
        similar_users = {}
        for other_user_id in self.user_item_matrix.index:
            if other_user_id != user_id:
                similarity = self.jaccard_similarity(user_id, other_user_id)
                similar_users[other_user_id] = similarity
        top_n_similar_users = sorted(similar_users.items(), key=lambda x: x[1], reverse=True)[:n]
        return top_n_similar_users

    def predict_rating(self, user_id, movie_id):
        similar_users = self.similar_users[user_id]

        numerator = 0
        denominator = 0
        for other_user_id, similarity in similar_users:
            if movie_id in self.user_item_matrix.columns:
                rating = self.user_item_matrix.loc[other_user_id, movie_id]
                if rating > 0:
                    numerator += similarity * rating
                    denominator += similarity

        if denominator == 0:
            return 0
        
        return numerator / denominator

    def predict(self, data):
        preds = Parallel(n_jobs=-1)(delayed(self.predict_rating)(row.userId, row.movieId) for row in tqdm(data.itertuples(), total=len(data), desc="Predicting"))
        data['score'] = preds
        rmse = np.sqrt(((data['rating'] - data['score']) ** 2).mean())
        return rmse

    def save_predictions(self, data, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path, index=False)

    def recommend(self, num_recommendations=20):
        recommendations = Parallel(n_jobs=-1)(delayed(self.recommend_for_user)(user_id, num_recommendations) for user_id in tqdm(self.user_item_matrix.index, desc="Generating Recommendations"))
        recommendations = {user_id: recs for user_id, recs in recommendations}
        return recommendations

    def recommend_for_user(self, user_id, num_recommendations):
        similar_users = self.similar_users[user_id]
        
        candidate_movies = set()
        for other_user_id, _ in similar_users:
            candidate_movies.update(self.train[(self.train['userId'] == other_user_id)]['movieId'])
            
        user_movie_ids = self.train[self.train['userId'] == user_id]['movieId'].tolist()
        candidate_movies = [movie_id for movie_id in candidate_movies if movie_id not in user_movie_ids]
        
        movie_scores = []
        for movie_id in candidate_movies:
            score = self.predict_rating(user_id, movie_id)
            movie_scores.append((movie_id, score))
        
        top_movies = sorted(movie_scores, key=lambda x: x[1], reverse=True)[:num_recommendations]
        return user_id, [movie_id for movie_id, _ in top_movies]


def ucf(train, test, save_dir="data/ucf", top_k=50):
    model = ucf_model(train, top_k=top_k)
    train_rmse = model.predict(train)
    test_rmse = model.predict(test)
    model.save_predictions(train, f"{save_dir}/train_predictions.csv")
    model.save_predictions(test, f"{save_dir}/test_predictions.csv")
    recommendations = model.recommend(save_dir, num_recommendations=20)
    print(f"train_rmse: {train_rmse}\n test_rmse: {test_rmse}")
    return train_rmse, test_rmse, recommendations
