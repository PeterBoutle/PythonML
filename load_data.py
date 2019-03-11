import numpy as np
import pandas as pd

filepath = "data\\weight-height.csv"

data = pd.read_csv(filepath)
conversion = {'Male':1,'Female':0}
data["Gender"] = data["Gender"].map(conversion)
data = data.to_numpy()
#shuffle the data around so that we can take the first x percent better.
data = np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)
train_size = 0.75
test_size = 0.25

train_rows = int(len(data) * train_size)
test_rows = int(len(data) * test_size)

train = data[:train_rows]
test = data[train_rows:train_rows + test_rows]


print("train consists of {} records:".format(len(train)))
print(train)

print("test consists of {} records:".format(len(test)))
print(test)

