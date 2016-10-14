
## 这里我们利用体征数据对Iris鲜花分类数据进行研究，由于其含有多项自变量
## 需要预处理，我们通过Pipeline 模块进行整合，简化流程。

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

## 准备Iris鲜花数据
iris = load_iris()
X, y = iris.data, iris.target

""" 对数据进行预处理
"""
## 我们发现大量自变量是高度相关的，所以用主成分分析的方法提取最显著的两
## 个主成分进行预测。
pca = PCA(n_components=2)

## 同时我们也希望用已有的变量直接进行预测，这里我们选取预测效果最好的一
## 个进行预测。
selection = SelectKBest(k=1)

## 这里我们将前面的变量整合起来
combined_features = FeatureUnion([("pca", pca), 
                                  ("univ_select", selection)])

## 这里是最后起到分类作用的SVM 分类器模块
svm = SVC(kernel="linear")

## 最后把所有模块整合起来，形成一个pipeline对象
pipeline = Pipeline([("features", combined_features), ("svm", svm)])

## 对pipeline对象进行训练
pipeline.fit(X,y)

## 利用训练好的pipeline对象进行预测
pipeline.predict(X) 

