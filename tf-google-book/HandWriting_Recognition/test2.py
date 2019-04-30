import numpy as np


def read_data():
  data = np.load("mnist.npz")
  x_train = data["x_train"]
  y_train = data["y_train"]
  crops = data["crops"]
  return x_train,y_train,crops

x, y, c = read_data()
s = dict()
for i in range(60000):
  hashed = hash(x[i,:,:].tostring())
  if hashed in s:
    print("!!")
  s[hashed] = y[i]

pic = np.zeros((28,28), dtype=np.uint8)
for i1 in range(16):
  for i2 in range(16):
    for i3 in range(16):
      for i4 in range(16):
        if len(set([i1, i2, i3, i4])) != 4:
          continue
        pic[:14,:14] = c[i1]
        pic[:14,14:] = c[i3]
        pic[14:, :14] = c[i2]
        pic[14:, 14:] = c[i4]
        hashed = hash(pic.tostring())
        if hashed in s:
          print(s[hashed])