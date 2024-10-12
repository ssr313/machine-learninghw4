import pandas as pd

train_data = pd.read_csv('train1_icu_data.csv')
train_labels = pd.read_csv('train1_icu_label.csv')

test_data = pd.read_csv('test1_icu_data.csv')
test_labels = pd.read_csv('test1_icu_label.csv')
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# 数据标准化
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# 设置不同的参数
params = [
    {'C': 1, 'kernel': 'linear'},
    {'C': 0.1, 'kernel': 'linear'},
    {'C': 1, 'kernel': 'rbf'},
    {'C': 1, 'kernel': 'sigmoid'}
    # 添加更多参数组合
]

# 训练和评估模型
for param in params:
    svm = SVC(**param)
    svm.fit(train_data_scaled, train_labels['hospital_death'])
    scores = cross_val_score(svm, train_data_scaled, train_labels['hospital_death'], cv=5)
    print(f"Parameters: {param}, Cross-validation scores: {scores.mean()}",'\t')