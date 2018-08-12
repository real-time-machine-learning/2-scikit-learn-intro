# 这里我们用逻辑回归来判断手写数字的扫描图像来判断数字是多少。

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 导入数据
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

n_samples = len(X_digits)

# 拆分训练集和测试集

# 这里因为我们的测试集和训练集不存在时间先后关系，所以可以使用Scikit
# Learn自带的 train_test_split 函数自动化拆分数据集

X_train, X_test, y_train, y_test = train_test_split(
    X_digits,
    y_digits,
    test_size=0.1)

model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 进行预测
prediction = model.predict(X_test)

score = model.score(X_test, y_test)
print(score)
