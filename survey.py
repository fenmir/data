# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 12:32:36 2018

@author: ST20942
"""

import numpy as np
import pandas as pd
from ggplot import *
import os

# Read the file
os.chdir(r'C:\Users\ST20942\Documents\work\171220 survey')
data = pd.read_csv('surveyResult.csv', encoding = 'Shift_JISx0213')

# Find X and y
X = data.iloc[:, 3:38]
y = data.iloc[:, 2]

questions = X.columns
X.columns = list(range(len(questions)))

# Remove redundant texts in the answers
columns = X.columns
for column in columns:
    not_null = ~X[column].isnull()
    X[column][not_null] = X[column][not_null].str.replace(r'_.*', '').astype(int)

# Fill nan with mean
X = X.fillna(X.mean())

# Split the data into test, train sets
np.random.seed(0)
index = np.random.permutation(len(X))
test_ratio = int(len(X) * 0.2)

X_train = X.iloc[index[:-test_ratio]]
y_train = y.iloc[index[:-test_ratio]]
X_test = X.iloc[index[-test_ratio:]]
y_test = y.iloc[index[-test_ratio:]]

# K-means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(random_state = 0, n_clusters = 3).fit(X)
label = kmeans.labels_

# Visualization using PCA with K-means labeling
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

x_min, x_max = X.iloc[:, 0].min() - .5, X.iloc[:, 0].max() + .5
y_min, y_max = X.iloc[:, 1].min() - .5, X.iloc[:, 1].max() + .5

plt.figure(1, figsize=(8,6))
plt.clf()

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(X)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c = label,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

# Visualization using LDA





# linear regression
from sklearn import linear_model

linearReg = linear_model.LinearRegression()
linearReg.fit(X_train, y_train)

scores = dict()
scores['linear'] = linearReg.score(X_test, y_test)

# Top/worst five questions whose the biggest/smallest coefficient
# The best questions are mostly about game characters and the worst are about battle contents.
top_five = linearReg.coef_.argsort()[-5:]
worst_five = linearReg.coef_.argsort()[:5]

print('レベルへ最もPositiveな影響を与えたベスト５質問 :')
for i, item in enumerate(questions[top_five]):
    print( str(i + 1), '. ' + item, sep = '')

print('')

print('レベルへ最もネガティブな影響を与えたワースト５質問 :')
for i, item in enumerate(questions[worst_five]):
    print( str(i + 1), '. ' + item, sep = '')

'''
レベルへ最もPositiveな影響を与えたベスト５質問 :
1. リトルナイツの戦闘（攻撃戦・防御戦）については面白いと感じましたか？
2. キャラクターや背景、マップなどデザインコンセプトは全体的に調和し、魅力的に感じましたか？
3. キャラクターカードをアップグレード(レベルアップ)する方法について分かりやすかったですか？
4. 相手の攻撃・防御ユニットの内容を考慮してキャラクター配置を考えることが出来ましたか？
5. リトルナイツを遊び続ける事で自分の村とキャラクターを成長させたいと感じましたか？

レベルへ最もネガティブな影響を与えたワースト５質問 :
1. ゲーム内で「HELP」と表示されている際、助けてあげたいと思いましたか？
2. ストーリーモードを通じて、ストーリーとバトルについて理解できたと思いますか？
3. バトルでは戦術を適用することによって様々なゲームプレイが可能だと思いましたか？
4. ゲームUIは直感的で分かりやすかったですか？（※ UI…ユーザーインターフェースの事を指します）
5. 自分の持っているキャラを使って自分だけのデッキを作るのは楽しいと感じましたか？
'''

# Linear - feature selection?



# Visualization of the questions based on the k-mean labels
# 각 그릅별로 GG 플롯 facet 으로 주요 질문에 대한 분포 비교해 보기
# R 에서 그린 전체 질문 내용 수직으로 나열한 거 그리기
