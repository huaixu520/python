import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1、
data = pd.read_json('./News.json', lines=True, encoding='utf-8')
# 2、
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
data['category'].value_counts().plot.pie(autopct='%1.1f%%')
plt.show()
news = data['headline'] + data['short_description']

texts = ",".join(news)
l_text = texts.lower()  # 全部转化为小写以方便处理
# 4、
df_text = []  # 将新闻内容放入列表中
j = 1
for i in news:
    df_text.append(i)
    j = j + 1
    if j == 5001:
        break
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(' '), stop_words=[''])  # 创建词袋数据结构
transformer = TfidfTransformer()
X = transformer.fit_transform(vectorizer.fit_transform(df_text)).toarray()
News_Category = set(data["category"].values)
News_Category = sorted(list(News_Category))
num = [x for x in range(0, 41)]
di = dict(zip(News_Category, num))
data["category"] = data["category"].map(di)
Y = np.array(data.iloc[0:5000, 0:1]).reshape(5000, )
print("数值化后得到的矩阵是：")
print(Y)
# 5、
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.75)
model = LogisticRegression()
model.fit(X_train, Y_train)
a = model.intercept_
b = model.coef_
print("截距:", a, " ", "参数:", b)
print("模型精度：", model.score(X_test, Y_test))
