import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras import models, layers
import matplotlib.pyplot as plt
import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
df = pd.read_csv('deep learning/neural networks/data/housepricedata.csv')
print(df.head())
data = df.values

X = data[:,0:10]
y = data[:,10]

MMScaler = preprocessing.MinMaxScaler()
X = MMScaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5)

model = models.Sequential(
    [
       layers.Dense(32, activation='relu', input_shape=(10,)),
       layers.Dense(32, activation='relu'),
       layers.Dense(1, activation='sigmoid'),
    ]
)

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_valid, y_valid))
accuracy = model.evaluate(X_test, y_test)[1]
print(accuracy)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()