from pycaret.datasets import get_data
from pycaret.classification import *

# 載入 Titanic 數據集
titanic = get_data('titanic')

# 設置 PyCaret 環境
exp = setup(
    data=titanic,
    target='Survived',
    session_id=123,           # 確保結果可重現
    normalize=True,           # 標準化數據
    polynomial_features=True, # 創建多項式特徵
    remove_multicollinearity=True, # 移除高度相關的特徵
    multicollinearity_threshold=0.95, # 多重共線性閾值
    fold=5,  # 使用交叉驗證來評估模型
    verbose=True
)

# 比較所有模型並選擇最佳模型（以 AUC 排序）
best_model_before_tuning = compare_models(
    sort='AUC',  # 以 AUC 指標排序
    n_select=1,  # 選擇最佳模型
    exclude=['KNeighborsClassifier']  # 排除 KNeighborsClassifier 模型
)

# 拉取比較結果表格
results_before_tuning = pull()

# 優化最佳模型
tuned_model = tune_model(best_model_before_tuning)

# 拉取優化後的模型結果表格
results_after_tuning = pull()

# 計算準確率
accuracy_before_tuning = results_before_tuning.loc[results_before_tuning.index[0], 'Accuracy']  # 第一行為最佳模型
accuracy_after_tuning = results_after_tuning.loc[results_after_tuning.index[0], 'Accuracy']    # 優化後的模型結果

# 顯示結果
print(f"最佳模型（未優化）：{best_model_before_tuning}")
print(f"最佳模型準確率（未優化）：{accuracy_before_tuning:.4f}")
print(f"最佳模型準確率（優化後）：{accuracy_after_tuning:.4f}")
