import tensorflow as tf
import numpy as np
import pandas as pd

# tf.logging.set_verbosity(tf.logging.ERROR)

cols = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
dataset = pd.read_csv('iris.data', names=cols)
dataset['class'] = [0] * 50 + [1] * 50 + [2] * 50

training_data = pd.concat([dataset[:40], dataset[50:90], dataset[100:140]]).sample(frac=1).reset_index(drop=True)
test_data = pd.concat([dataset[40:50], dataset[90:100], dataset[140:]]).sample(frac=1).reset_index(drop=True)

training_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(training_data[cols].values, tf.float32),
            tf.cast(training_data['class'].values, tf.int32)
        )
    )
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(test_data[cols].values, tf.float32),
            tf.cast(test_data['class'].values, tf.int32)
        )
    )
)

model = tf.keras.Sequential([
	tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4, )),
	tf.keras.layers.Dense(3, activation=tf.nn.softmax)
	])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
