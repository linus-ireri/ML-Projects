import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_iris()

x = data.data
y = data.target

x_train, x_test, y_train,y_test = train_test_split(
    x,y, test_size=0.2, random_state=42)

model = RandomForestClassifier()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: ", accuracy)