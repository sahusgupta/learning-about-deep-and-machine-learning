from keras import datasets, layers, models, losses
import tensorflow as tf, sys, io
import matplotlib.pyplot as plt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

fashion_mnist = datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = models.Sequential(
    [layers.Flatten(input_shape=(28, 28)),
    layers.Dense(784, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')]
)

model.compile(optimizer='adam', loss= losses.SparseCategoricalCrossentropy, metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

pred_model = models.Sequential([model, layers.Softmax()])
t_pred = pred_model.predict(x_test[:3])
preds = pred_model.predict(x_test)
print(t_pred)

labels = ['T-shirt', 'Pants', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

plt.figure(figsize=(10,5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    pred_label = labels[tf.argmax(preds[i]).numpy()]
    real = labels[y_test[i]]
    if pred_label == real:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel(f"{pred_label} ({real})", color=color)
    plt.show()

