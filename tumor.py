
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt



# %%
import os

# %%
def create_folder(x):
    if os.path.exists(x):
        print("Folder already exists !!!")

    else:
        os.mkdir(x)
        print("{x} created Successfully !!!")

# %%
folderpath_Train = "//app//train"
folderpath_Test = "//app//test"
folderpath_Validate = "//app//validate"

create_folder(folderpath_Train)
create_folder(folderpath_Test)
create_folder(folderpath_Validate)

# %%
folder_path_train_damaged = "//app//train//damaged"
folder_path_train_good = "//app//train//good"
folder_path_test_damaged = "//app//test//damaged"
folder_path_test_good = "//app//test//good"
folder_path_validate_damaged = "//app//validate//damaged"
folder_path_validate_good = "//app//validate//good"



create_folder(folder_path_train_damaged)
create_folder(folder_path_train_good)

create_folder(folder_path_test_damaged)
create_folder(folder_path_test_good)

create_folder(folder_path_validate_good)
create_folder(folder_path_validate_damaged)




from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=folderpath_Train,target_size=(224,224), batch_size=10,classes=['good','damaged'])

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=folderpath_Test,target_size=(224,224), batch_size=10,classes=['good','damaged'])

validate_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=folderpath_Validate,target_size=(224,224), batch_size=10,classes=['good','damaged'])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

model = Sequential([ Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding='same',input_shape=(224,224,3)),
        MaxPool2D(pool_size=(2,2), strides=2),
        Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding ='same'),
        MaxPool2D(pool_size=(2,2), strides=2),
        Flatten(),
        Dense(units=2, activation = 'softmax')])

# %%
model.summary()

# %%
model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=train_batches,validation_data=validate_batches, epochs=10,verbose=1)

# %%
pred = model.predict(test_batches)

# %%
from sklearn.metrics import confusion_matrix
import numpy as np

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(pred, axis = -1))

# %%
cm

# %%
accuracy = (cm[0][0] + cm[1][1])/(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
print(accuracy)

# %% [markdown]
# Plotting Learning Curve

# %%
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.show()

# %%
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.show()


print("working")


model.save("/models/tumor_model.h5")