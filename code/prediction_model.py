import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, linear_model
from sklearn.metrics import accuracy_score
import lightgbm as lgb

inputfile = '../features/cleaned_prediction_data.xls'

data = pd.read_excel(inputfile)


train = data.drop(['Customer_type'], axis=1)
target = data['Customer_type']

# 訓練資料70%，測試資料30%
# random_state = 0，每次數據都一樣
train_x,test_x, train_y, test_y = train_test_split(train, target, test_size = 0.3, random_state = 0)

# 建立羅吉斯回歸模型
logistic_regr = linear_model.LogisticRegression()
logistic_regr.fit(train_x, train_y)

# 計算準確率
y_pred = logistic_regr.predict(test_x)
accuracy = accuracy_score(test_y, y_pred)
print('logistic regression accuracy: ',accuracy)


# 建立LightGBM模型
gbm0 = lgb.LGBMClassifier(
    num_leaves=15,
    learning_rate=0.01,
    n_estimators=700,
    num_class=3 # 分類數目
)

gbm0.fit(train_x, train_y)
y_pred = gbm0.predict(test_x)
accuracy = accuracy_score(test_y, y_pred)
print('lgbm accuracy: ',accuracy)