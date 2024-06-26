import pandas as pd
import matplotlib.pyplot as plt
import os
import csv

# ファイルの読み込み
# 現在のワーキングディレクトリを取得
current_dir = os.getcwd()
print(current_dir)
target_dir = os.path.join(current_dir, 'results')
files=os.listdir(target_dir)
accurucy_list = []
k_list = []
#k_train=1000は今回だけ、今後はファイル名に記入すること
k_num=1000
for file in files:
    if not file.endswith('.csv'):
        continue
    #csvファイルを読み込む
    df = pd.read_csv(target_dir + '\\' + file,encoding_errors='ignore')
    #セルの値を取得
    df=df[df.iloc[:, 0] == "Accuracy"]
    accurucy_list.append(df.iloc[0,2])
    #print(df)
    k_list.append(k_num)
    k_num += 100

#グラフを描画
plt.plot(k_list, accurucy_list, marker='o')
plt.title('K-NN_accuracy')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.grid(True)

# グラフをファイルとして保存
plt.savefig('K-NN_accuracy_data.png')

# グラフを表示（必要なら）
plt.show()