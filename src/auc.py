import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from ucf import ucf
# 读取CSV文件
df = pd.read_csv('data/ratings_small.csv')

# 随机打乱数据 
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 使用 train_test_split 将数据分成二八两份
train, test = train_test_split(df_shuffled, test_size=0.2, random_state=42)

# 保存分割后的数据到新的CSV文件
train.to_csv('data/train_data.csv', index=False)
test.to_csv('data/test_data.csv', index=False)

train["label"] = train["rating"]
test["label"] = test["rating"]

ucf(train, test)
exit(0)
# 计算AUC
def calculate_auc(df, num_classes):
    y_true = df['label']
    y_score = df['score']
    auc_list = []

    # One-vs-Rest (OvR) approach for multi-class AUC
    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(int)
        auc = roc_auc_score(y_true_binary, y_score)
        auc_list.append(auc)
    
    return np.mean(auc_list)

num_classes = 6  # 0-5

auc_train = calculate_auc(train, num_classes)
auc_test = calculate_auc(test, num_classes)

print(f"AUC on Train Data: {auc_train}")
print(f"AUC on Test Data: {auc_test}")