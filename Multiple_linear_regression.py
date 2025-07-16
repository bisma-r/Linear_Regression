from sklearn.datasets import fetch_california_housing
import numpy as np

data = fetch_california_housing(as_frame=True)
df = data.frame

x = np.array(df.drop(columns=['MedHouseVal']).values)
y = np.array(df['MedHouseVal'].values)

class MLR:
    def __init__(self, x, y, a=0.01):
        self.x = x.copy()
        self.y = y
        self.m, self.n = x.shape  # m = number of samples, n = number of features
        self.w = np.ones(self.n)
        self.b = 0
        self.a = a
        self.j = float('inf')  # initial cost set as infinity
        self.mean = None
        self.sd = None

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
            if count % 100 == 0 or count == 1:
                print(f"Iter {count}: cost = {self.j:.6f}, b = {self.b:.4f}, w[0] = {self.w[0]:.4f}")
        print(f"Training completed in {count} iterations.")

    def normalize(self):
        self.mean = np.mean(self.x, axis=0)
        self.sd = np.std(self.x, axis=0)
        # Avoid division by zero
        self.sd[self.sd == 0] = 1
        self.x = (self.x - self.mean) / self.sd


mlr = MLR(x, y)
mlr.run()
