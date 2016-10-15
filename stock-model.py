
"""《实时机器学习实战》　彭河森、汪涵

机器学习工具Scikit Learn介绍

这里我们将利用Scikit Learn对苹果公司的秒级股票报价进行建模，并以此为例
熟悉Scikit Learn的各个模块。
""" 

import pandas as pd 

"""数据导入

这里我们导入经过了预处理的苹果公司股票报价，该数据仅包含正常交易时间的
数据。
"""

data = pd.read_csv("aapl-trading-hour.csv",
                   index_col = 0)

"""数据整理 

Scikit learn 的preprocessing 模块目前只能将X, Y形态的数据整理成为需要的
形似。这里我们需要对时间序列数据进行重拍，将其整理成为X, y形式。

对于每一个观测，因变量y代表当前秒的变化率，自变量X代表前几秒的变化率和
成交量变化情况。

这里有两个办法进行这样的处理：

1) 通过for 循环，将观测逐个加入到X, y矩阵和向量中。这种方法短平快，但是
程序可重复利用率较低。

2) 撰写preprocessing 模块，通过preprocessing 模块的fit/transform模式对
数据进行转换。这样的方法看似麻烦，但是代码可重复利用率高，可以为后面操
作节省很多工作。

这里我们将会采用两种方法进行比较。
""" 

def embed_time_series(x, k):
    """this function would transform an N dimensional time series into a
    tuple containing: 

    1) an (n - k) by k matrix that is [X[i], x[i+1], ... x[i+k-1]],
    for i from 0 to n-k-1
    
    2) a vector of length (n - k) that is [x[k], x[k+1] ... x[n]]
    """
    n = len(x)

    if k >= n: 
        raise "Can not deal with k greater than the length of x" 
    
    output_y = list(x[k:])
    output_x = list(map(lambda i: list(x[i:(i+k)]), 
                        range(0, n-k)))
    return (output_x, output_y)

class TimeSeriesEmbedder(): 
    def __init__(self, k):
        self.k = k 
    def fit(self):
        pass
    def transform(self, X, y):
        return embed_time_series(X, self.k)
        
