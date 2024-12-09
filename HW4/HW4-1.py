# 載入必要的庫
import pandas as pd
from pycaret.classification import *
from pycaret.datasets import get_data

# 載入 Titanic 數據集
titanic = get_data('titanic')

# 數據預處理
titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])

# 設置 PyCaret 環境
clf1 = setup(
    data=titanic,
    target='Survived',
    session_id=123,
    normalize=True,
    categorical_features=['Sex', 'Embarked'],
    html=False  # 禁用互動式 HTML 小部件
)

# 比較分類模型
best_model = compare_models(
    include=[
        'lightgbm', 'catboost', 'xgboost', 'rf', 'et', 'gbc', 'ada', 
        'knn', 'svm', 'ridge', 'qda', 'lda', 'nb', 'dt', 'dummy', 'lr'
    ],
    n_select=16
)

print(best_model)
