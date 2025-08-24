#Predict pass/fail based on student scores (tiny CSV)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import pandas as pd

# fetch dataset
student_performance = fetch_ucirepo(id=320)

# data (as pandas dataframes)
X = student_performance.data.features.copy()
y = student_performance.data.targets.copy()

y = y['G3'].apply(lambda g: 'Pass' if g >= 10 else 'Fail')
X = X.drop(columns=['G3'])
X = pd.get_dummies(X)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
model = RandomForestClassifier(random_state= 0)
model.fit(train_X, train_y)

val_predictions = model.predict(val_X)
print("Accuracy: ",accuracy_score(val_y, val_predictions))