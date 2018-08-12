# 这里我们用K-近邻估计来判断手写数字的扫描图像来判断数字是多少。

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 导入数据
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

n_samples = len(X_digits)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_digits,
    y_digits,
    test_size=0.1)

model = KNeighborsClassifier()

# 训练模型
model.fit(X_train, y_train)

# 进行预测
prediction = model.predict(X_test)

score = model.score(X_test, y_test)
print(score)
