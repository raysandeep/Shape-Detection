import cv2, numpy as np, os
from keras.utils import to_categorical
import datetime

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Dataset dir not output dir
dir = '/home/sandy/Desktop/sandeep/dataset2/'
os.chdir(dir)
img_size = 60 



folders, labels, images = ['triangle', 'star', 'square', 'circle'], [], []
for folder in folders:
    for path in os.listdir(os.getcwd()+'/'+folder):
        img = cv2.imread(folder+'/'+path,0)
        #cv2.imshow('img', img)
        #cv2.waitKey(1)
        images.append(cv2.resize(img, (img_size, img_size)))
        labels.append(folders.index(folder))
    
print("[ EXTRACT COMPLETED ]")
to_train= 0
train_images, test_images, train_labels, test_labels = [],[],[],[]
for image, label in zip(images, labels):
    if to_train<5:
        train_images.append(image)
        train_labels.append(label)
        to_train+=1
    else:
        test_images.append(image)
        test_labels.append(label)
        to_train = 0


dataDim = np.prod(images[0].shape)
img = np.array(train_images).reshape(len(train_images), dataDim).astype('float32')
img /=255
train_data = img
img = np.array(test_images).reshape(len(test_images), dataDim).astype('float32')
img /=255
test_data = img
#print(train_data)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)


classes = len(np.unique(train_labels))

model = Sequential()
model.add(Dense(256, activation = 'tanh', input_shape = (dataDim,)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(classes, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, batch_size = 256, epochs=50, verbose=1,validation_data=(test_data, test_labels_one_hot))

[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print(test_acc)
x = datetime.datetime.now().strftime("%X").split(":")
x = str(x[0])+"_"+str(x[1])+"_"+str(x[2])
#model.save('/home/sandy/Desktop/sandeep/model/model'+x+'.h5')
print('[ COMPLETED MODEL CREATION SUCCESFULLY ] with name ' + 'model'+x+'.h5')