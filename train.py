import matplotlib.pyplot as plt
import os
import cv2
import glob
from keras.utils import to_categorical
from tqdm import tqdm, tnrange
from keras.callbacks import ModelCheckpoint
from keras.optimizers import rmsprop
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from model import *
import numpy as np
from skimage.transform import resize


def draw_training_curve(history):
    plt.figure(1)
    # History for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig("accuracy.png")
    # History for loss
    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
    plt.savefig("loss.png")

if __name__ == '__main__':
    # Command line parameters
    model = model(img_shape=(50, 50, 1))
    model.summary()

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])
    filepath = "/home/ajey/Desktop/weights/weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Get data
    ids = os.listdir("/home/ajey/Desktop/previous/")
    X1 = np.zeros((len(ids), 50, 50, 1), dtype=np.float32)
    X2 = np.zeros((len(ids), 50, 50, 1), dtype=np.float32)
    X3 = np.zeros((len(ids), 50, 50, 1), dtype=np.float32)
    stop = []
    next = []
    previous = []
    # tqdm is used to display the progress bar
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        # Load images
        path_images1 = '/home/ajey/Desktop/next/'
        path_images2 = '/home/ajey/Desktop/previous/'
        path_images3 = '/home/ajey/Desktop/stop/'

        img1 = load_img(path_images1+id_, color_mode="grayscale")
        x1_img = img_to_array(img1)

        img2 = load_img(path_images2+id_, color_mode="grayscale")
        x2_img = img_to_array(img2)

        img3 = load_img(path_images3+id_, color_mode="grayscale")
        x3_img = img_to_array(img3)

        X1[n] = x1_img/255.0
        next.append('0')
        X2[n] = x2_img/255.0
        previous.append('1')
        X3[n] = x3_img/255.0
        stop.append('2')
    data = np.concatenate((X1, X2, X3), 0)
    labels = next+previous+stop
    labels = to_categorical(labels, num_classes=3, dtype='float32')

    # Split train and valid 90-10
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.10, shuffle=True)

    # Train
    print('[INFO] Training the model')
    history = model.fit(train_data, np.array(train_labels), validation_split=0.10, epochs=5,
                        batch_size=32, callbacks=callbacks_list, verbose=1)

    # Evaluate the model
    print('[INFO] Evaluating the trained model...')
    (loss, accuracy) = model.evaluate(test_data, np.array(test_labels),
                                      batch_size=32,
                                      verbose=1)
    print('[INFO] accuracy: {:.2f}%'.format(accuracy * 100))

    # Visualize training history

    draw_training_curve(history)
