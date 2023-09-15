import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 加载数据集
data = pd.read_csv("creditcard.csv")

# 分割特征和标签
X = data.drop(["Class"], axis=1)
y = data["Class"]

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

C_normal_as_fraud = 1
C_fraud_as_normal = 10
# 创建随机森林模型
model = RandomForestClassifier(random_state=42)
# 定义成本矩阵（根据金额和错分情况调整）
cost_matrix = [[0, C_normal_as_fraud],
               [C_fraud_as_normal, 0]]
# 计算样本权重（基于成本矩阵和交易金额）
sample_weights = []
for amount in X_train["Amount"]:
    if y_train == 0:
        weight = np.log(amount + 1)  # 这里使用对数函数作为示例，你可以根据需求选择不同的权重计算方式
        weight = cost_matrix[1][0] * weight  # 欺诈交易被错误分类为正常的代价
    else:
        weight = cost_matrix[0][1]  # 正常交易被错误分类为欺诈的代价
    sample_weights.append(weight)

# 训练模型时使用样本权重
model.fit(X_train, y_train, sample_weight=sample_weights)

# 预测
y_pred = model.predict(X_test)

# 评估模型
print(classification_report(y_test, y_pred))
