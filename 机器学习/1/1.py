import pandas as pd
from sklearn import model_selection

data = pd.read_csv('Iris.csv')
data.replace("Iris-setosa", 0, inplace=True)
data.replace("Iris-versicolor", 1, inplace=True)
data.replace("Iris-virginica", 2, inplace=True)

trainingSet, testSet = model_selection.train_test_split(data, test_size=0.7)

print(trainingSet)
print(testSet)
