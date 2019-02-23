import pandas as pd

datafile = '../dataset/air_data.csv'  # 原始數據
resultfile = '../features/explore.xls'  # 數據探索結果表

data = pd.read_csv(datafile, engine='python')

# percentiles：賦值類似列表形式，可選，表示百分位數，介於0和1之間。
# 默認值為 [.25,.5,.75]，分別返回第25，第50和第75百分位數，可自定義其它值。
# include='all'，輸入的所有行都將包含在輸出中。
# T是轉置，轉置後更方便查閱。
# https://zhuanlan.zhihu.com/p/56526297
explore = data.describe(percentiles=[], include='all').T
print(explore)

# describe()函數可計算非空值數，len()可得到數據長度。
# 相減即為missing value數量
explore['null'] = len(data) - explore['count']

# 只選取部分探索結果
explore = explore[['null', 'max', 'min']]
explore.columns = ['空值數', '最大值', '最小值']  # 表頭重命名

explore.to_excel(resultfile)  # 導出結果