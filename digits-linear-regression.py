
## 这里我们用逻辑回归来判断手写数字的扫描图像来判断数字是多少。

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

## 导入数据
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

n_samples = len(X_digits)

## 拆分训练集和测试集
X_train = X_digits[:.9 * n_samples]
y_train = y_digits[:.9 * n_samples]
X_test = X_digits[.9 * n_samples:]
y_test = y_digits[.9 * n_samples:]

model = LogisticRegression()

## 训练模型
model.fit(X_train, y_train)

## 进行预测
prediction = model.predict(X_test)

score = model.score(X_test, y_test)
print(score) 
