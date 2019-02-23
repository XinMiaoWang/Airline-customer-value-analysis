import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from cluster_plot import print_cluster_result, plot_cluster # 外部檔案

inputfile = '../features/data_cleaned_for_kmeans.xls'  # 待聚類的數據資料

k = 5  # KMeans要分的類別數，需結合業務的理解和分析來確定客戶的類別數量

data = pd.read_excel(inputfile)

# 調用k-means算法，進行聚類分析
# n_jobs是並行數，一般等於CPU數較好
kmodel = KMeans(n_clusters=k, n_jobs=1)
kmodel.fit(data)  # 訓練模型

print_cluster_result(data, kmodel)
plot_cluster(data, kmodel)

