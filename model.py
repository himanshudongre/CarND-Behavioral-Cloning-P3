import numpy as np
import csv
from scipy import ndimage
from keras import backend as K
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import MaxPooling2D, Conv2D, Dropout, Flatten, Dense, Cropping2D, Lambda
import matplotlib.pyplot as plt

#dir_path = "../../../opt/carnd_p3/data"
dir_path = "/home/workspace/CarND-Behavioral-Cloning-P3/Recordings"
lines = []

with open(dir_path + "/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    #next(reader, None)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    #Include center ,left and right images and apply correction to treat them as center camera image
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = dir_path + '/IMG/' + filename
        image = ndimage.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        if i ==0:
            correction = 0
        elif i == 1:
            correction = 0.2
        else:
            correction = -0.2

        measurement = measurement+correction
        measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

def yuv_conversion(x):
    return K.tf.image.rgb_to_yuv(x)

def resize(x):
    return K.tf.image.resize_images(x, (66,200))

#Model
def Custom_Nvidia_Model():

    # defining our model
    model = Sequential()

    #Crop unwanted part of images
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

    #Resize image
    model.add(Lambda(resize, input_shape=(None,None,3)))

    #Convert image to YUV colorspace
    model.add(Lambda(yuv_conversion, input_shape=(66, 200, 3)))

    #lambda layer to normalize and mean center images
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(66,200,3)))

    # 1 conv2D layer=> input: 66x200x3, total_params: 5x5x24x3+24=1824
    model.add(Conv2D(24, (5, 5), strides=(2,2), input_shape=(66, 200, 3), activation='relu'))

    # 2 conv2D layer=> input: 31x98x24, total_params: 5x5x36x24+36=21636
    model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))

    # 3 conv2D layer=> input: 14x47x36, total_params: 5x5x48x36+48=43248
    model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))

    # 4 conv2D layer=> input: 5x22x48, total_params=3x3x64x48+64=27712
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # 5 conv2D layer=> input: 3x20x64, total_params: 3x3x64x64+64=36928, output_shape=(1x18x64)
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # Dropout=> total_params=0, output_shape=1x18x64
    model.add(Dropout(0.5))

    # Flatten=> total_params=0, output_shape: 1152
    model.add(Flatten())

    # Dense=> output_shape=100, total_params=1152x100+100=115300
    model.add(Dense(100, activation='relu'))

    # Dropout=> total_params=0, output_shape=100
    model.add(Dropout(0.5))

    # Dense=> total_params=100x50+50=5050
    model.add(Dense(50, activation='relu'))

    # Dense=> total_params=50x10+10=510
    model.add(Dense(10, activation='relu'))

    # Dense=> total_params=10x1+1=11
    model.add(Dense(1))

    # compile
    opti=Adam(lr=0.001)
    model.compile(loss='mse', optimizer=opti)
    return model

# intiialize the model
model = Custom_Nvidia_Model()

# train the model
history = model.fit(X_train, y_train,validation_split = 0.2 , epochs=3, batch_size=64, shuffle=True)

# save the trained model which can be used to run the drive.py file
model.save("model.h5")

# viewing the plots for accuracy and loss
# Loss
plt.plot(history.history['loss']);
plt.plot(history.history['val_loss']);
plt.legend(['Training','Validation']);
plt.title('Loss');
plt.xlabel('Epoch');

# Accuracy
plt.plot(history.history['acc']);
plt.plot(history.history['val_acc']);
plt.legend(['Training','Validation']);
plt.title('Accuracy');
plt.xlabel('Epoch');
