import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 業務理解 (Business Understanding)
# 本次任務是預測鐵達尼號乘客的生還情況，並基於此結果提交預測檔案。

# 2. 數據理解 (Data Understanding)
# 加載數據
train_data = pd.read_csv('./HW2/train.csv')
test_data = pd.read_csv('./HW2/test.csv')

# 檢查數據結構和摘要統計
print(train_data.info())
print(train_data.describe())

# 3. 數據準備 (Data Preparation)
# 填補缺失值
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# 轉換類別特徵為數值特徵
le = LabelEncoder()
train_data['Embarked'] = le.fit_transform(train_data['Embarked'])
test_data['Embarked'] = le.transform(test_data['Embarked'])
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

# 選擇我們的特徵
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_data[features]
y = train_data['Survived']

# 分割訓練和驗證集（80% 訓練，20% 驗證）
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 建模 (Modeling)
# 初始化隨機森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 預測驗證集
y_pred = model.predict(X_valid)

# 評估模型準確率
accuracy = accuracy_score(y_valid, y_pred)
print(f'初始模型準確率: {accuracy:.2f}')

# 顯示混淆矩陣
cm = confusion_matrix(y_valid, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Overall Confusion Matrix for All Features')
# plt.show()

# 詳細分類報告
print(classification_report(y_valid, y_pred))

# 特徵重要性可視化
importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# 繪製特徵重要性圖
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances')
# plt.show()

# 5. 評估 (Evaluation)
# 定義參數網格進行網格搜索
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 最佳參數
print("最佳參數: ", grid_search.best_params_)

# 使用最佳參數進行預測
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_valid)

# 混淆矩陣及可視化 (最佳模型)
cm_best = confusion_matrix(y_valid, y_pred_best)
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix with Best Parameters')
# plt.show()

# 新的準確率
accuracy_best = accuracy_score(y_valid, y_pred_best)
print(f'最佳模型準確率: {accuracy_best:.2f}')

# 特徵分析：性別與生存率
sns.barplot(data=train_data, x='Sex', y='Survived', hue='Pclass')
plt.title('Survival Rate by Gender and Class')
# plt.show()

# 特徵分析：年齡與生存率
sns.histplot(train_data[train_data['Survived'] == 1]['Age'], color='blue', kde=True, label='Survived')
sns.histplot(train_data[train_data['Survived'] == 0]['Age'], color='red', kde=True, label='Not Survived')
plt.legend()
plt.title('Age Distribution by Survival')
# plt.show()

# 6. 部署 (Deployment)
# 使用最佳模型對測試集進行預測
test_predictions = best_model.predict(test_data[features])

# 創建提交文件
submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": test_predictions
})
submission.to_csv('./HW2/submission.csv', index=False)
