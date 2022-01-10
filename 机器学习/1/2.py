import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("College.csv")
data_new = data.drop(["Unnamed: 0"], axis=1)
desc = data_new.describe(include='all')  # include='all',代表对所有列进行统计，如果不加这个参数，则只对数值列进行统计
print(desc)

sns.boxplot(x='Private', y='Outstate', data=data_new)
plt.ylabel('Private', fontsize=15.0)
plt.xlabel('Outstate', fontsize=15.0)
plt.yticks(fontsize=15.0)
plt.show()
