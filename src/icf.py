import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
import json

class icf_model:
    def __init__(self, train, top_k):
        self.train = train
        self.top_k = top_k
        self.user_item_matrix = train.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        self.similar_items = self.precompute_similar_items()

    def precompute_similar_items(self):
        item_ids = self.user_item_matrix.columns
        item_vectors = self.user_item_matrix.T.values
        similarities = cosine_similarity(item_vectors)
        
        similar_items = {}
        results = Parallel(n_jobs=-1)(delayed(self.get_top_n_similar_items)(item_id, similarities[idx], self.top_k)
                                      for idx, item_id in tqdm(enumerate(item_ids), total=len(item_ids), desc="Processing Similar Items"))
        
        for item_id, sim_items in results:
            similar_items[item_id] = sim_items
        
        return similar_items

    def get_top_n_similar_items(self, item_id, item_similarities, n):
        similar_items = sorted(
            enumerate(item_similarities),
            key=lambda x: x[1],
            reverse=True
        )
        top_n_similar_items = [
            (self.user_item_matrix.columns[idx], similarity)
            for idx, similarity in similar_items if self.user_item_matrix.columns[idx] != item_id
        ][:n]
        return item_id, top_n_similar_items

    def predict_rating(self, user_id, item_id):
        if item_id not in self.similar_items:
            return 0

        similar_items = self.similar_items[item_id]

        numerator = 0
        denominator = 0
        for other_item_id, similarity in similar_items:
            if other_item_id in self.user_item_matrix.columns:
                rating = self.user_item_matrix.loc[user_id, other_item_id]
                if rating > 0:
                    numerator += similarity * rating
                    denominator += similarity

        if denominator == 0:
            return 0
        
        return numerator / denominator

    def predict(self, data):
        preds = []
        for row in tqdm(data.itertuples(), total=len(data), desc="Predicting"):
            preds.append(self.predict_rating(row.userId, row.movieId))
        data['score'] = preds
        rmse = np.sqrt(((data['rating'] - data['score']) ** 2).mean())
        return rmse

    def save_predictions(self, data, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path, index=False)

    def recommend(self, num_recommendations=20):
        # recommendations = Parallel(n_jobs=-1)(delayed(self.recommend_for_user)(user_id, num_recommendations) for user_id in tqdm(self.user_item_matrix.index, desc="Generating Recommendations"))
        recommendations = []
        for user_id in tqdm(self.user_item_matrix.index, desc="Generating Recommendations") :
            recommendations.append(self.recommend_for_user(user_id, num_recommendations))
        recommendations = {user_id: recs for user_id, recs in recommendations}
        return recommendations

    def recommend_for_user(self, user_id, num_recommendations):
        user_rated_items = set(self.train[self.train['userId'] == user_id]['movieId'])
        candidate_items = set()
        
        for item_id in user_rated_items:
            if item_id in self.similar_items:
                similar_items = self.similar_items[item_id]
                candidate_items.update(item_id for item_id, _ in similar_items)
        
        candidate_items -= user_rated_items
        
        item_scores = []
        for item_id in candidate_items:
            score = self.predict_rating(user_id, item_id)
            item_scores.append((item_id, score))
        
        top_items = sorted(item_scores, key=lambda x: x[1], reverse=True)[:num_recommendations]
        return user_id, [item_id for item_id, _ in top_items]

def icf(train, test, save_dir="data/icf", top_k=50):
    model = icf_model(train, top_k=top_k)
    train_rmse = model.predict(train)
    test_rmse = model.predict(test)
    model.save_predictions(train, f"{save_dir}/train_predictions.csv")
    model.save_predictions(test, f"{save_dir}/test_predictions.csv")
    # model.save_recommendations(save_dir, num_recommendations=20)
    recommendations = model.recommend(num_recommendations=20)
    print(f"train_rmse: {train_rmse}\n test_rmse: {test_rmse}\n")
    return train_rmse, test_rmse, recommendations

# 示例调用
# train = pd.read_csv("path_to_train.csv")
# test = pd.read_csv("path_to_test.csv")
# icf(train, test)
