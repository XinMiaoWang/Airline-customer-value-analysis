import pandas as pd
from datetime import datetime

# 字串轉日期
def string_to_date(dateString):
    return datetime.strptime(dateString, "%Y/%m/%d")

datafile = '../dataset/air_data.csv'  # 原始數據
cleanedfile_1 = '../features/data_cleaned_for_kmeans.xls'
cleanedfile_2 = '../features/data_cleaned_for_prediction.xls'

data = pd.read_csv(datafile, engine='python')

data = data[data['SUM_YR_1'].notnull() & data['SUM_YR_2'].notnull()]  # 非空值的票價才保留

# 只保留票價非零的，或者平均折扣率與總飛行公里數同時為0的記錄。
index1 = data['SUM_YR_1'] != 0
index2 = data['SUM_YR_2'] != 0
index3 = (data['SEG_KM_SUM'] == 0) & (data['avg_discount'] == 0)
data = data[index1 | index2 | index3]

# 格式轉換
data['LOAD_TIME'] = data['LOAD_TIME'].apply(string_to_date)
data['FFP_DATE'] = data['FFP_DATE'].apply(string_to_date)

data['MEMBER_TIME'] = (data['LOAD_TIME'] - data['FFP_DATE'])/30 # 成為會員多久了(月)
data['LAST_TO_END'] = round(data['LAST_TO_END']/30,2)# 距離最後一次乘坐的時間(月)

tmp_data = data[['MEMBER_TIME','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','avg_discount']]
LRFMC = tmp_data.describe().T
LRFMC = LRFMC[['max', 'min']]
print(LRFMC)

# z-score
tmp_data = (tmp_data - tmp_data.mean(axis=0)) / (tmp_data.std(axis=0))


tmp_data.to_excel(cleanedfile_1, index=False)
data.to_excel(cleanedfile_2, index=False)