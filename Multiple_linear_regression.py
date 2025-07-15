from sklearn.datasets import fetch_california_housing
import numpy as np

data = fetch_california_housing(as_frame=True)
df = data.frame

x = np.array(df.drop(columns=['MedHouseVal']).values)
y = np.array(df['MedHouseVal'].values)

class MLR:
    def __init__(self, x, y, a=0.01):
        self.x = x
        self.y = y
        self.m, self.n = x.shape  # m = number of samples, n = number of features
        self.w = np.ones(self.n)
        self.b = 1
        self.a = a
        self.j = float('inf')  # initial cost set as infinity

    def cost_function(self):
        predictions = np.dot(self.x, self.w) + self.b
        errors = predictions - self.y
        cost = (1 / (2 * self.m)) * np.sum(errors ** 2)
        return cost

    def update_parameters(self):
        predictions = np.dot(self.x, self.w) + self.b
        errors = predictions - self.y

        dw = (1 / self.m) * np.dot(self.x.T, errors)
        db = (1 / self.m) * np.sum(errors)

        self.w -= self.a * dw
        self.b -= self.a * db

    def run(self, max_iter=10000, tolerance=1e-6):
        self.normalize()
        count = 0
        while self.j > tolerance and count < max_iter:
            self.j = self.cost_function()
            self.update_parameters()
            count += 1
            #if count % 100 == 0 or count == 1:
            print(f"Iter {count}: cost = {self.j:.6f}, b = {self.b:.4f}, w[0] = {self.w[0]:.4f}")
        print(f"Training completed in {count} iterations.")

    def normalize(self):
        s = [0] * self.n
        s1 = [0] * self.n
        mean = [0] * self.n
        sd = [0] * self.n

        for row in self.x:
            for i, val in enumerate(row):
                s[i] += val

        # Compute mean for each feature
        for i in range(self.n):
            mean[i] = s[i] / self.m

        # Compute standard deviation for each feature
        for row in self.x:
            for i, val in enumerate(row):
                s1[i] += (val - mean[i]) ** 2

        for i in range(self.n):
            sd[i] = (s1[i] / self.m) ** 0.5

        for row in self.x:
            for i in range(self.n):
                if sd[i] != 0:
                    row[i] = (row[i] - mean[i]) / sd[i]
                else:
                    row[i] = 0


mlr = MLR(x, y)
mlr.run()
