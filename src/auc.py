import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from ucf import ucf
from icf import icf
from dnn import dnn
from sklearn.preprocessing import label_binarize
import os
import json
from utils import load_movie_mapping, convert_movieid_to_imdbid, convert
# 读取CSV文件
df = pd.read_csv('data/ratings_small.csv')

# 随机打乱数据 
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 使用 train_test_split 将数据分成二八两份
train, test = train_test_split(df_shuffled, test_size=0.2, random_state=42)

labels = sorted(train['rating'].unique())

movieid_to_imdbid = load_movie_mapping("data/links_small.csv")


data = {}

# 保存分割后的数据到新的CSV文件
train.to_csv('data/train_data.csv', index=False)
test.to_csv('data/test_data.csv', index=False)

train["label"] = train["rating"].apply(lambda x: 1 if x >= 4 else 0)
test["label"] = test["rating"].apply(lambda x: 1 if x >= 4 else 0)


# save_dir="data/ucf_top200"
# train_rmse, test_rmse, recommendations = ucf(train, test, save_dir, top_k=200)

# save_dir="data/icf_top200"
# train_rmse, test_rmse, recommendations, similar_items = icf(train, test, save_dir, top_k=200)

save_dir="data/dnn3"
train_rmse, test_rmse, recommendations, aucs = dnn(train, test, save_dir=save_dir, epoch=4, lr=5e-5)
data["train_aucs"] = aucs


# tmp = {}
# for x,y in similar_items.items() :
#     x = convert(x, mapping_dict=movieid_to_imdbid)
#     if x == -1 :
#         continue
#     y = [pair[0] for pair in y]
#     tmp[x] = y
# similar_items = tmp
# similar_items = convert_movieid_to_imdbid(similar_items, movieid_to_imdbid)
# recs_df = pd.DataFrame([
#             {'tmdbId': tmdbId, 'tmdbIds': movie_ids}
#             for tmdbId, movie_ids in similar_items.items()
#         ])
# recs_df.to_csv(f"{save_dir}/similar_movies.csv", index=False)

recommendations = convert_movieid_to_imdbid(recommendations, movieid_to_imdbid)
recs_df = pd.DataFrame([
            {'userId': user_id, 'tmdbIds': movie_ids}
            for user_id, movie_ids in recommendations.items()
        ])
recs_df.to_csv(f"{save_dir}/recommendations.csv", index=False)
# exit(0)

# exit(0)
# 计算AUC
def calculate_auc(df):
    y_true = df['label']
    y_score = df['score']
    auc = roc_auc_score(y_true, y_score)
    
    return auc

num_classes = 10  # 0-5

auc_train = calculate_auc(train)
auc_test = calculate_auc(test)

print(f"AUC on Train Data: {auc_train}")
print(f"AUC on Test Data: {auc_test}")

data["train_rmse"] = train_rmse,
data["test_rmse"] = test_rmse,
data["auc_train"] = auc_train,
data["auc_test"] = auc_test


os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, "log.json")
with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)