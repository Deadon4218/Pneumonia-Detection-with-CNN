
import os
import numpy as np
import cv2
from numpy import random
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model
from matplotlib import pyplot
import matplotlib.patheffects as path_effects

input_path = './input/chest_xray/'
#defined some constants for later usage
img_dims = 160
epochs = 15
batch_size = 32
class_names = ['/NORMAL/', '/PNEUMONIA/']


#Fitting the CNN to the images
def process_data(img_dims, batch_size):
    # Data generation objects
    train_datagen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.3, shear_range=0.2, vertical_flip=True)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_directory(
        directory=input_path + 'train',
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)

    val_gen = val_datagen.flow_from_directory(
        directory=input_path + 'val',
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)

    # I will be making predictions off of the test set in one batch size
    # This is useful to be able to get the confusion matrix
    test_data = []
    test_labels = []
    label = 0
    for cond in ['/NORMAL/', '/PNEUMONIA/']:
        for img in tqdm(os.listdir(input_path + 'test' + cond)):
            img = plt.imread(input_path + 'test' + cond + img)
            img = cv2.resize(img, (img_dims, img_dims))
            if len(img.shape) == 1:
                img = np.dstack([img, img, img])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255
            if cond == '/NORMAL/':
                label = 0
            elif cond == '/PNEUMONIA/':
                label = 1
            test_data.append(img)
            test_labels.append(label)

    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    return train_gen, val_gen, test_data, test_labels

train_gen, val_gen, test_data, test_labels = process_data(img_dims, batch_size)

#Initialising the CNN
model = models.Sequential()

model.add(layers.Conv2D(16,(3,3),activation = 'relu', padding='same',input_shape=(img_dims,img_dims,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(32,(3,3),activation = 'relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation = 'relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation = 'relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(256,(3,3),activation = 'relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Dropout(0.3))

model.add(layers.Flatten())
# Creating 1 Dense Layer
model.add(layers.Dense(units=128,activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(units=1, activation='sigmoid'))



#Compiling the CNN
model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])

#model.load_weights('best_weights.hdf5')

checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=2, mode='max')
# early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')

hist = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=[checkpoint, lr_reduce])

test_loss, test_acc = model.evaluate(test_data, test_labels)

##Predictions
predictions = model.predict(test_data)
predictions = np.array(predictions).reshape(-1)

##Accuracy
print("Untrained model, accuracy: {:5.2f}%".format(100 * test_acc))

acc = accuracy_score(test_labels, np.round(predictions))*100
cm = confusion_matrix(test_labels, np.round(predictions))
tn, fp, fn, tp = cm.ravel()

print('CONFUSION MATRIX ------------------')
print(cm)

print('\nTEST METRICS ----------------------')
precision = tp/(tp+fp)*100
recall = tp/(tp+fn)*100
print('Accuracy: {}%'.format(acc))
print('Precision: {}%'.format(precision))
print('Recall: {}%'.format(recall))
print('F1-score: {}'.format(2*precision*recall/(precision+recall)))

#Visualize the accuracy plots and the model loss
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax = ax.ravel()

for i, met in enumerate(['accuracy', 'loss']):
   ax[i].plot(hist.history[met])
   ax[i].plot(hist.history['val_' + met])
   ax[i].set_title('Model {}'.format(met))
   ax[i].set_xlabel('epochs')
   ax[i].set_ylabel(met)
   ax[i].legend(['train', 'val'])
plt.show()

##Visualize the predicted x-ray images from the test
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = (0 if (predictions_array[i] < 0.5) else 1)
    if int(predicted_label) == int(true_label):
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:.3f} ({})".format(class_names[predicted_label],
                                       predictions_array[i],
                                       true_label), color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(2))
    plt.yticks([])
    thisplot = plt.bar(range(2), [1 - predictions_array[i], predictions_array[i]], color="#777777")
    plt.ylim([0, 1])
    predicted_label = (0 if (predictions_array[i] < 0.5) else 1)

    if predicted_label == int(true_label):
        thisplot[int(true_label)].set_color('blue')
    else:
        thisplot[predicted_label].set_color('red')


def data_test_opt(pred):
    pred[pred > 0.5] = int(1)
    pred[pred <= 0.5] = int(0)

    return np.array(pred).reshape(-1)

##Shuffling the x-ray images
def shuffle_tow_array(a, b):
    indices = np.random.permutation(len(a))
    A = a[indices]
    B = b[indices]
    return A, B

test_data, test_labels = shuffle_tow_array(test_data, test_labels)

################################################################################################################


num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
ind = random.randint(len(predictions), size=(num_images))

for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(ind[i], predictions, test_labels, test_data)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(ind[i], predictions, test_labels)
plt.tight_layout()
plt.show()



##print the names of the layers from Feature map and from the filters
def showFiltersBlocks():
    indexFilter = []
    for index in range(len(model.layers)):
        layer = model.layers[index]
        # check for convolutional layer
        if 'conv' not in layer.name:
            continue
        # get filter weights
        filters, biases = layer.get_weights()
        indexFilter.append(index)
        print(layer.name, filters.shape)
    print(indexFilter)
    return indexFilter

print("_____________________Visualize Feature Map Start_____________________")
##Visualize Feature Map
def showFeatureMap(indexs):
    outputs = [model.layers[i].output for i in indexs]
    model2 = Model(inputs=model.inputs, outputs=outputs)
    # get feature map for first hidden layer
    feature_maps = model2.predict(test_data[1:6])
    # plot the output from each block
    # plot all 64 maps in an 8x8 squares
    plt.figure(figsize=(15, 15))
    j = 0
    for fmap in feature_maps:
        square = 4
        ix = 1
        fig = plt.figure(figsize=(5, 5))
        filters, biases = model.layers[indexs[j]].get_weights()
        print(model.layers[indexs[j]].name)
        text = fig.text(0.5, 1., str(model.layers[indexs[j]].name) + "  " + str(filters.shape),
                        ha='center', va='center', size=20)
        text.set_path_effects([path_effects.Normal()])
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(fmap[0, :, :, ix - 1])
                ix += 1
        j += 1
    # show the figure
    plt.show()


showFeatureMap(showFiltersBlocks())

print("_____________________Visualize Filters Start_____________________")


##Visualize Filters
def showFilters(indexs):
    for index in indexs:
        fig = plt.figure(figsize=(5, 5))
        filters, biases = model.layers[index].get_weights()
        text = fig.text(0.5, 1., str(model.layers[index].name) + "  " + str(filters.shape),
                        ha='center', va='center', size=20)
        text.set_path_effects([path_effects.Normal()])
        # normalize filter values to 0-1 so we can visualize them
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        # plot first few filters
        n_filters, ix = 6, 1
        for i in range(n_filters):
            # get the filter
            f = filters[:, :, :, i]
            # plot each channel separately
            for j in range(3):
                # specify subplot and turn of axis
                ax = pyplot.subplot(n_filters, 3, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                pyplot.imshow(f[:, :, j])
                ix += 1
        # show the figure
        pyplot.show()


showFilters(showFiltersBlocks())
########################################################