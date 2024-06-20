import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class ucf_model:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.user_item_matrix = train.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    def jaccard_similarity(self, user1, user2):
        user1_movies = set(self.train[(self.train['userId'] == user1) & (self.train['rating'] >= 4)]['movieId'])
        user2_movies = set(self.train[(self.train['userId'] == user2) & (self.train['rating'] >= 4)]['movieId'])
        
        intersection = len(user1_movies.intersection(user2_movies))
        union = len(user1_movies.union(user2_movies))
        
        if union == 0:
            return 0
        
        return intersection / union

    def get_similar_users(self, user_id):
        similar_users = {}
        for other_user_id in self.user_item_matrix.index:
            if other_user_id != user_id:
                similarity = self.jaccard_similarity(user_id, other_user_id)
                similar_users[other_user_id] = similarity
        return similar_users

    def predict_rating(self, user_id, movie_id):
        similar_users = self.get_similar_users(user_id)
        similar_users = sorted(similar_users.items(), key=lambda x: x[1], reverse=True)[:10]  # 选取最相似的10个用户

        numerator = 0
        for other_user_id, similarity in similar_users:
            if movie_id in self.user_item_matrix.columns:
                rating = self.user_item_matrix.loc[other_user_id, movie_id]
                if rating > 0:
                    numerator += similarity * rating
                # else : print(f"user_id: {other_user_id}, movie_id: {movie_id} , rating: {rating}")

        return numerator 
    
    def predict(self):
        preds = []
        for row in tqdm(self.test.itertuples(), total=len(self.test), desc="Predicting"):
            preds.append(self.predict_rating(row.userId, row.movieId))
        self.test['score'] = preds

        rmse = np.sqrt(((self.test['rating'] - self.test['score']) ** 2).mean())
        return rmse

def ucf(train, test, save_dir="data/ucf"):
    model = ucf_model(train, test)
    return model.predict()
