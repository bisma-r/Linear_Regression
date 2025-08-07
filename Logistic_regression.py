from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

# Load the dataset
data = load_breast_cancer()

# Convert to pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

x = np.array(df.drop(columns=['target']).values)
y = np.array(df['target'].values)

class Logistic_Regression:
    def __init__(self, x, y, a = 0.01, l = 1000):
        self.x = x.copy()
        self.y = y
        self.m, self.n = x.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.a = a
        self.l = l
        self.j = float('inf')
        self.mean = None
        self.sd = None

    def cost_function(self):
        epsilon = 1e-15
        weighted_sum = np.dot(self.x, self.w) + self.b
        predictions = 1 / (1 + np.exp(-weighted_sum))
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        cost = (-1/self.m) * np.sum((self.y * np.log(predictions)) + (1 - self.y) * (np.log(1 - predictions))) + ((self.l / (2 * self.m)) * np.sum(self.w ** 2))
        return cost

    def update_parameters(self):
        weighted_sum = np.dot(self.x, self.w) + self.b
        predictions = 1 / (1 + np.exp(-weighted_sum))
        errors = predictions - self.y

        dw = (1 / self.m) * (np.dot(self.x.T, errors)) + ((self.l/self.m) * self.w)
        db = (1 / self.m) * np.sum(errors)

        self.w -= self.a * dw
        self.b -= self.a * db

    def normalize(self):
        self.mean = np.mean(self.x, axis = 0)
        self.sd = np.std(self.x, axis = 0)
        self.sd[self.sd == 0] = 1
        self.x = (self.x - self.mean) / self.sd

    def run(self, max_iter = 10000, tolerance = 1e-6):
        self.normalize()
        count = 0
        while self.j > tolerance and count < max_iter:
            self.j = self.cost_function()
            self.update_parameters()
            count +=1
            print (f"Iter {count}: cost = {self.j: .6f}, b = {self.b: .4f}, w[0] = {self.w[0]: .4f}")
        print(f"Training completed in {count} iterations.")

lr = Logistic_Regression(x, y)
lr.run()

