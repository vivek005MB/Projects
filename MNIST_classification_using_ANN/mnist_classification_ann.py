# -*- coding: utf-8 -*-
"""p1_MNIST_classification_ANN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1H_2QbDTUdY8Qu5xwSt2_Iu7ZmVUMOwWm
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist

# unpacking
(train_data,train_labels),(test_data,test_labels) =mnist.load_data()

# Let's see the training data
print(f"Training sample :\n{train_data[0]}\n")
print(f"Training label :\n{train_labels[0]}\n")

train_data.shape,train_labels.shape,test_data.shape,test_labels.shape

train_data[0].shape,train_labels[0].shape

# plot a single image
import matplotlib.pyplot as plt
i=7
plt.imshow(train_data[i])
print(f"label is {train_labels[i]}")

class_names = ['0','1','2','3','4','5','6','7','8','9']
len(class_names)

plt.imshow(train_data[1],cmap = plt.cm.binary)
plt.title(class_names[train_labels[1]])

import random
plt.figure(figsize = (10,7))
for i in range(6):
    ax = plt.subplot(2,3,i+1)
    rand_index = random.choice(range(len(train_data)))
    plt.imshow(train_data[rand_index],cmap=plt.cm.binary)
    plt.title(class_names[train_labels[rand_index]])
    plt.axis(False)

"""# Model|"""

tf.random.set_seed(48)

# Create
model_1 = tf.keras.Sequential()
model_1.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model_1.add(tf.keras.layers.Dense(4,activation='relu'))
model_1.add(tf.keras.layers.Dense(4,activation='relu'))
model_1.add(tf.keras.layers.Dense(10,activation='softmax'))

model_1.summary()

# Compile
model_1.compile(
loss = tf.keras.losses.SparseCategoricalCrossentropy(),
optimizer = tf.keras.optimizers.Adam(),
metrics = ['accuracy']
)

# Fit
history_not_norm = model_1.fit(
train_data,train_labels,epochs = 10,
validation_data = (test_data,test_labels)    
)

"""### after 10 epochs :
the accuracy i got is : 88.72% on training set and 77.86% on test set

"""

# Let's improve the model
train_data.min(),train_data.max()

# Normalizing the data
train_data = train_data / train_data.max()
test_data = test_data / test_data.max()
train_data.min(),train_data.max()

"""## Model 2"""

tf.random.set_seed(48)
model_2 = tf.keras.Sequential([
tf.keras.layers.Flatten(input_shape=(28,28)),
tf.keras.layers.Dense(4,activation='relu'),
tf.keras.layers.Dense(4,activation='relu'),
tf.keras.layers.Dense(10,activation='softmax')                               
])
model_2.compile(
loss = tf.keras.losses.SparseCategoricalCrossentropy(),
optimizer = tf.keras.optimizers.Adam(),
metrics = ['accuracy']    
)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch:1e-3 * 10 **(epoch/20))
norm_history = model_2.fit(
train_data,
train_labels,
epochs = 50,
validation_data = (test_data,test_labels),
callbacks = [lr_scheduler]    
)

import pandas as pd
# Plot non-normalized data loss curves
pd.DataFrame(history_not_norm.history).plot(title="Non-normalized Data")
# Plot normalized data loss curves
pd.DataFrame(norm_history.history).plot(title="Normalized data");

import numpy as np 
import matplotlib.pyplot as plt
lrs = 1e-3 * (10 * (np.arange(50)/20))
plt.plot(lrs,norm_history.history['loss'])
plt.xlabel("Learning rate")
plt.ylabel('Loss')
plt.title('Finding the ideal learning rate')

"""## Model_3"""

tf.random.set_seed(48)
model_3 = tf.keras.Sequential([
tf.keras.layers.Flatten(input_shape=(28,28)),
tf.keras.layers.Dense(4,activation='relu'),
tf.keras.layers.Dense(4,activation='relu'),
tf.keras.layers.Dense(10,activation='softmax')                               
])
model_3.compile(
loss = tf.keras.losses.SparseCategoricalCrossentropy(),
optimizer = tf.keras.optimizers.Adam(lr=0.005),
metrics = ['accuracy']    
)
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch:1e-3 * 10 **(epoch/20))
history = model_3.fit(
train_data,
train_labels,
epochs = 50,
validation_data = (test_data,test_labels)
#,callbacks = [lr_scheduler]    
)

# Make predictions with the most recent model
y_probs = model_3.predict(test_data) # "probs" is short for probabilities

# View the first 5 predictions
y_probs[:5]

# See the predicted class number and label for the first example
y_probs[0].argmax(), class_names[y_probs[0].argmax()]

# Convert all of the predictions from probabilities to labels
y_preds = y_probs.argmax(axis=1)

# View the first 10 prediction labels
y_preds[:10]

# Check out the non-prettified confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true=test_labels, 
                 y_pred=y_preds)

!git clone https://github.com/vivek005MB/HelperCodes.git

!mv /content/HelperCodes/confusionMatrix.py /content

# from confusionMatrix import make_confusion_matrix
import confusionMatrix

import numpy as np
# Make a prettier confusion matrix
confusionMatrix.make_confusion_matrix(y_true=test_labels, 
                      y_pred=y_preds,
                      classes=class_names,
                      figsize=(15, 15),
                      text_size=10)

classes = class_names
true_labels = test_labels
plt.figure(figsize=(7,7))
for j in range(6):
    i = random.randint(0, len(test_data))
    ax = plt.subplot(2,3,j+1)
   
    # Create predictions and targets
    target_image = test_data[i]
    pred_probs = model_3.predict(target_image.reshape(1, 28, 28)) # have to reshape to get into right size for model
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]
    plt.imshow(test_data[i],cmap=plt.cm.binary)
    plt.title(class_names[test_labels[i]])
    plt.axis(True)
    if pred_label == true_label:
        color = "green"
    else:
        color = "red"
    # Add xlabel information (prediction/true label)
    plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                   100*tf.reduce_max(pred_probs),
                                                   true_label),
             color=color) # set the color to green or red
