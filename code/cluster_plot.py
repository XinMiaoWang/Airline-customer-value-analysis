import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cleanedfile = '../features/air_prediction_data.xls'
originalfile = '../features/data_cleaned_for_prediction.xls'

original_data = pd.read_excel(originalfile)

def print_cluster_result(data, kmodel):    
    cluster_labels = kmodel.labels_ # 樣本聚類結果
    r1 = pd.Series(cluster_labels).value_counts()  # 統計各個類別的數目
    r2 = pd.DataFrame(kmodel.cluster_centers_)  # 找出聚類中心
    r = pd.concat([r2, r1], axis=1)  # 橫向連接（0是縱向），得到聚類中心對應的類別下的數目
    r.columns = list(data.columns) + ['類別數目']  # 重新命名表頭
    print(r)

    # print(kmodel.cluster_centers_)  # 查看聚類中心
    # print('labels_=', kmodel.labels_)  # 查看各樣本對應的類別
	
    original_data['Customer_grouping'] = cluster_labels
    print(original_data.head())
    original_data.to_excel(cleanedfile, index=False)


def plot_cluster(data, kmodel):
    labels = data.columns  # 標籤
    k = 5  # 數據個數
    plot_data = kmodel.cluster_centers_
    color = ['b', 'g', 'r', 'c', 'y']  # 指定顏色

    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    plot_data = np.concatenate((plot_data, plot_data[:, [0]]), axis=1)  # 閉合
    angles = np.concatenate((angles, [angles[0]]))  # 閉合

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    for i in range(len(plot_data)):
        ax.plot(angles, plot_data[i], 'o-', color=color[i],
                label='客戶群' + str(i), linewidth=2)  # 畫線

    ax.set_rgrids(np.arange(0.01, 3.5, 0.5),
                  np.arange(-1, 2.5, 0.5), fontproperties="SimHei")
    ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties="SimHei")
    plt.legend(loc=4)
    plt.show()