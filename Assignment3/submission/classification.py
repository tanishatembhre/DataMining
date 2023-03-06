import argparse
import csv
import os
import shutil
import tempfile

import cv2
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.metrics import Precision, SensitivityAtSpecificity, SpecificityAtSensitivity, Accuracy
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 500, 580


def create_model(image_shape):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()

    model.add(Conv2D(32, (2, 2), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=["accuracy", Precision(), SensitivityAtSpecificity(0.5), SpecificityAtSensitivity(0.5)])
    return model


def data_generator(input_folder, batch_size, stage="train"):
    if stage == "train":
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    else:
        datagen = ImageDataGenerator(rescale=1. / 255)

    return datagen.flow_from_directory(input_folder,
                                       target_size=(img_width, img_height),
                                       batch_size=batch_size,
                                       class_mode='binary')


def train_model(model, data, data_size, batch_size=10, epochs=50):
    train_generator, validation_generator = data
    train_samples, validation_samples = data_size
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('model/save_at_{epoch}_{val_loss:03f}.keras', save_best_only=True, monitor='val_loss',
                               mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    callbacks = [earlyStopping, mcp_save, reduce_lr_loss]

    model.fit(
        train_generator,
        callbacks=callbacks,
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size)


def generate_label_folders(patient_folder, patient, rsn_folder, noise_folder, iclabel):
    for image in os.listdir(patient_folder):
        if image.endswith("thresh.png"):
            ic_number = image.split("_")[1]
            label = iclabel[ic_number]
            if label == 0:
                print(os.path.join(patient_folder, image), os.path.join(noise_folder, str(patient) + image))
                shutil.copyfile(os.path.join(patient_folder, image), os.path.join(noise_folder, str(patient) + image))
            else:
                shutil.copyfile(os.path.join(patient_folder, image), os.path.join(rsn_folder, str(patient) + image))


# def correct_files(idir):
#     for image in os.listdir(idir):
#         img = cv2.imread(os.path.join(idir, image))
#         img = img[20:, :img.shape[1] - 75, :]
#         cv2.imwrite(os.path.join(idir, image), img)


def generate_train_test_data(data_folder, train_dir, test_dir, ):
    train_rsn_dir = os.path.join(train_dir, "RSN")
    train_noise_dir = os.path.join(train_dir, "Noise")
    test_rsn_dir = os.path.join(test_dir, "RSN")
    test_noise_dir = os.path.join(test_dir, "Noise")
    dirs = [train_rsn_dir, test_rsn_dir, train_noise_dir, test_noise_dir]
    for dir in dirs:
        try:
            shutil.rmtree(dir)   
        except:
            pass
        os.makedirs(dir)
    patients = [fname.split("_")[1] for fname in os.listdir(data_folder) if fname.endswith(".csv")]
    test_patient = patients[-1]
    train_patients = patients[:-1]
    for patient in train_patients:
        iclabel = gen_ic_dict(data_folder, patient)
        patient_folder = os.path.join(data_folder, f"Patient_{patient}")
        generate_label_folders(patient_folder, patient, train_rsn_dir, train_noise_dir, iclabel)

    for patient in test_patient:
        iclabel = gen_ic_dict(data_folder, patient)
        patient_folder = os.path.join(data_folder, f"Patient_{patient}")
        generate_label_folders(patient_folder, patient, test_rsn_dir, test_noise_dir, iclabel)

    # for dir in dirs:
    #     correct_files(dir)
    train_size = count_files(train_rsn_dir) + count_files(train_noise_dir)
    test_size = count_files(test_rsn_dir) + count_files(test_noise_dir)
    return train_size, test_size


def count_files(folder):
    return len(os.listdir(folder))


def gen_ic_dict(data_folder, patient):
    iclabel = {}
    with open(os.path.join(data_folder, f'Patient_{patient}_Labels.csv'), 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            iclabel[row['IC']] = int(row['Label'])
    return iclabel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datafolder", type=str, help="data folder for training", required=True)
    parser.add_argument("-e", "--epochs", type=int, help="epochs for training", required=False, default=50)
    parser.add_argument("-b", "--batchsize", type=int, help="batch size for training", required=False, default=10)
    args = parser.parse_args()
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_size, test_size = generate_train_test_data(args.datafolder, train_dir, test_dir)

    train_ds = data_generator(train_dir, args.batchsize)
    val_ds = data_generator(test_dir,batch_size =args.batchsize, stage="val")

    model = create_model((img_width, img_height))
    train_model(model, (train_ds, val_ds), (train_size, test_size))
