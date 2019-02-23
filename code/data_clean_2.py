import numpy as np
import pandas as pd


inputfile = '../features/air_prediction_data.xls'
outputfile = '../features/cleaned_prediction_data.xls'

data = pd.read_excel(inputfile)  # 讀取數據

# 使用到的特徵
use_data = data[['FFP_TIER','AVG_INTERVAL','avg_discount','EXCHANGE_COUNT','Eli_Add_Point_Sum',
                 'SUM_YR_1','SUM_YR_2','P1Y_BP_SUM','L1Y_BP_SUM','P1Y_Flight_Count','L1Y_Flight_Count',
                 'Customer_grouping']]

# 定義客戶類型(流失、準流失、未流失)
# 已流失客戶:第二年飛行次數與第一年飛行次數比例小於50%的客戶
# 準流失客戶:第二年飛行次數與第一年飛行次數比例在[50%,90%)內的客戶
# 未流失客戶:第二年飛行次數與第一年飛行次數比例大於90%的客戶
proportion = use_data['L1Y_Flight_Count']/use_data['P1Y_Flight_Count']
conds = [ proportion < 0.5, proportion < 0.9]
choices = ['0', '1']
use_data['Customer_type'] = np.select(conds, choices, default='2')
print(use_data.head())

use_data.to_excel(outputfile, index=False)