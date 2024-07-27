from model import SimpleLinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

model = SimpleLinearRegression()

data = pd.read_csv("data/Salary_dataset.csv")

x_train, x_test, y_train, y_test = train_test_split(data['YearsExperience'], data['Salary'], test_size=0.2, train_size=0.8, random_state=10)

model.fit(x_train, y_train)
print(model._coefs())

preds = model.predict(x_test)
print(model.err(y_test))
