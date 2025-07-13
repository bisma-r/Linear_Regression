from sklearn.datasets import fetch_california_housing
import numpy as np

data = fetch_california_housing(as_frame=True)
df = data.frame

x = np.array(df.drop(columns=['MedHouseVal']).values.tolist())
y = np.array(df['MedHouseVal'].tolist())

class MLR:
    def __init__(self, w, m = len(x),  b = 1, a = 0.01, d = 0, j = 2):
        self.w = w
        self.b = b
        self.m = m
        self.d = d
        self.a = a
        self.j = j

    def cost_function(self):
        self.j = (1 / (2 * self.m)) * self.sum_function()
        return self.j

    def sum_function(self, e = 0, b1 = 0):
        s = 0
        c = 0
        self.w = np.array(self.w)
        if self.d == 0:
            for i in x:
                s += ((np.dot(self.w, i) + self.b - y[c]) ** 2)
                c = c + 1
        elif self.d == 1:
            for i in x:
                s += ((np.dot(self.w, i) + b1 - y[c]) * i[e])
                c = c + 1
        else:
            for i in x:
                s += (np.dot(self.w, i) + self.b - y[c])
                c = c + 1
        return s

    def update_para(self):
        temp = self.b
        self.d = 2
        self.b = self.b - (self.a * ((1 / self.m) * self.sum_function()))
        self.d = 1
        for index, i in enumerate(self.w):
            i = i - (self.a * ((1 / self.m) * self.sum_function(index, temp)))
        return self.w, self.b


# max_iter = 10000 #chatGPT recommendation
# count = 0
# while j > 1e-6 and count < max_iter:
#
#     j = cost_function(w, b, m)
#     w, b = update_para(w, b, a, m)
#     count +=1
#     print ("w = ", w, " b = ", b, " j = ", j )