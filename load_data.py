import numpy as np

filepath = "data\\weight-height.csv"


data = np.genfromtxt(filepath, delimiter=',',names=True,dtype=None, encoding="utf8",usecols=(0,1,2))
#shuffle the data around so that we can take the first x percent better.
data = np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)
train_size = 0.75
test_size = 0.25

print(data.shape)

if data[:,0]=="Male":
    data[:,0]=1
else:
    data[:,0]=0

print(data)

train_rows = int(data.size * train_size)
test_rows = int(data.size * test_size)

train = data[:train_rows]
test = data[train_rows:train_rows + test_rows]



print(train.size)
print("test size {}".format(test.size))