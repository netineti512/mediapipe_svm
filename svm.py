import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, fbeta_score

y_true = []
for j in range(1, 5):
    for i in range(1, 61):
        y_true.append(j)

# csvファイルの読み込み
npArray = np.loadtxt("./train/trainD.csv", delimiter = ",", dtype = "float", skiprows = 1)

# 説明変数の格納
x = npArray[:, 2:]

# 目的変数の格納
y = npArray[:, 0:1].ravel()

# 学習手法にSVMを選択
model = svm.SVC(gamma=0.01, C=10., kernel='rbf')

# 学習
model.fit(x,y)

y_pred = []
for i in range(0,240):
    test = np.loadtxt("./test/testD.csv", delimiter = ",", dtype = "float")
    df = test[i:i+1]
    ans = model.predict(df)

    if ans == 1:
        print(i+1,int(ans),  "fuck")
        y_pred.append(int(ans))
    elif ans == 2:
        print(i+1,int(ans),  "good")
        y_pred.append(int(ans))
    elif ans == 3:
        print(i+1,int(ans),  "silent")
        y_pred.append(int(ans))
    elif ans == 4:
        print(i+1,int(ans),  "up")
        y_pred.append(int(ans))
    else:
        print(i+1,int(ans),  "不明")
        y_pred.append(int(ans))

tp = np.sum((np.array(y_true)==1)&(np.array(y_pred)==1))
tn = np.sum((np.array(y_true)==0)&(np.array(y_pred)==0))
fp = np.sum((np.array(y_true)==0)&(np.array(y_pred)==1))
fn = np.sum((np.array(y_true)==1)&(np.array(y_pred)==0))

accuracy_score = accuracy_score(y_true, y_pred)
error_rate = 1 - accuracy_score

print("正答率(予測がどれだけ正しいか)：", accuracy_score)
print("誤答率(予測がどれだけ誤っているか)：", error_rate)

print("tp:", tp)
print("tn:", tn)
print("fp:", fp)
print("fn:", fn)
