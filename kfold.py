import numpy as np # linear algebra
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import random
from sklearn.preprocessing import Binarizer
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from matplotlib import pyplot as plt
from keras import models

from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
#CNN
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import warnings
import os
from os import *
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import keras
from keras import backend as K
import tensorflow as tf
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.models import Model
from keras import optimizers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.models import load_model, save_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from PIL import ImageFile
warnings.simplefilter('error', Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000



datasetFolderName='C:/Users/user/Desktop/deeplearning/vgg16' # 파일 경로
MODEL_FILENAME="model_cv.h5" # 모델 이름
sourceFiles=[]
classLabels=['Ok', 'defect']


# source에 있는 data를 splitRate 비율만큼 dest로 옮긴다
def transferBetweenFolders(source, dest, splitRate):
    global sourceFiles
    sourceFiles = os.listdir(source)  # source 경로에 들어있는 파일명 반환

    if (len(sourceFiles) != 0):  # 파일이 0개가 아니면 실행되는 코드

        transferFileNumbers = int(len(sourceFiles) * splitRate)  # 지정한 비율만큼 나눈 파일 개수

        transferIndex = random.sample(range(0, len(sourceFiles)), transferFileNumbers)
        # 첫번째 매개 변수로 시퀀스 데이터 타입(튜플, 문자열, range, 리스트) 또는 set 타입
        # 두번째 매개 변수로는 랜덤하게 뽑을 인자의 개수
        # sample 함수는 첫번째 인자로 받은 시퀀스데이터 에서 두번째 매개변수개의 랜덤하고, unique하고, 순서상관없이 인자를 뽑아서 리스트로 만들어서 반환해줍니다.
        # 총 파일 개수에서 원하는 개수만 큼 랜덤하게 뽑아서 그 인덱스를 리스트로 반환

        for eachIndex in transferIndex:
            shutil.move(source + str(sourceFiles[eachIndex]), dest + str(sourceFiles[eachIndex]))
            # 특정경로에 있는 파일을 특정 경로로 이동

    else:
        print("No file moved. Source empty!")


def transferAllClassBetweenFolders(source, dest, splitRate):
    for label in classLabels:  # Ok , defect 폴더명

        transferBetweenFolders(datasetFolderName + '/' + source + '/' + label + '/',
                               datasetFolderName + '/' + dest + '/' + label + '/',
                               splitRate)


# First, check if test folder is empty or not, if not transfer all existing files to train
# test 폴더가 비어있지 않다면 test에 들어있는 모든 파일을 train으로 옮긴다
transferAllClassBetweenFolders('test', 'train', 1.0)


# Now, split some part of train data into the test folders.
# train에 있는 데이터의 20%를 test 폴더로 옮긴다.
transferAllClassBetweenFolders('train', 'test', 0.20)

X=[]
Y=[]


# train에 있는 데이터들을 x에 파일명 넣고 y에 label을 넣어준다
def prepareNameWithLabels(folderName):
    sourceFiles = os.listdir(datasetFolderName + '/train/' + folderName)  # Cs/train/Ok에 있는 파일명들을 list로 반환

    for val in sourceFiles:
        X.append(val)
        if (folderName == classLabels[0]):
            Y.append(0)
        #         elif(folderName==classLabels[1]):
        #             Y.append(1)
        else:
            Y.append(1)
#             Y.append(2)


prepareNameWithLabels(classLabels[0]) # Ok
prepareNameWithLabels(classLabels[1]) # defect
# prepareNameWithLabels(classLabels[2])

X=np.asarray(X)
Y=np.asarray(Y)

# learning rate
batch_size = 128
epoch= 50
activationFunction='relu'
drop = 0.5

rate = 0.000005

def getModel():



    input_tensor = Input(shape=(150,150,3))
    model = VGG16(weights='imagenet', include_top=False, input_tensor = input_tensor)

# model = VGG16(weights='imagenet', include_top=True)

# 모델 Layer 데이터화
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

# Layer 추가
    x = layer_dict['block5_pool'].output
# Cov2D Layer +
    x = Conv2D(filters = 64, kernel_size=(3, 3), activation='relu')(x)
# MaxPooling2D Layer +
    x = MaxPooling2D(pool_size=(2, 2))(x)
# Flatten Layer +
    x = Flatten()(x)
# FC Layer +
    x = Dense(2048, activation='relu')(x) # 2048 output
    x = Dropout(drop)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(drop)(x)
# 이 예제에서는 고양이와 강아지를 분류하는 신경망이여서 2개의 출력 softmax 함수로 구성 = Flatten()(x)

 # FC Layer +
    x = Dense(2048, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(drop)(x)
    # FC Layer +
    x = Dense(2048, activation='relu')(x)  # 2048 output
    x = Dropout(drop)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(drop)(x)
    # 이 예제에서는 고양이와 강아지를 분류하는 신경망이여서 2개의 출력 softmax 함수로 구성 = Flatten()(x)

    # FC Layer +
    x = Dense(2048, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(drop)(x)

    x = Dense(2048, activation='relu')(x)  # 2048 output
    x = Dropout(drop)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(drop)(x)
    # 이 예제에서는 고양이와 강아지를 분류하는 신경망이여서 2개의 출력 softmax 함수로 구성 = Flatten()(x)

    # FC Layer +
    x = Dense(2048, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(drop)(x)

    # 이 예제에서는 고양이와 강아지를 분류하는 신경망이여서 2개의 출력 softmax 함수로 구성
    x = Dense(2, activation='softmax')(x) #

# new model 정의
    new_model = Model(inputs = model.input, outputs = x)

# CNN Pre-trained 가중치를 그대로 사용할때
    for layer in new_model.layers[:19] :
        layer.trainable = False

    new_model.summary()

# #     # 컴파일 옵션
#     def focal_loss(y_true, y_pred):
#         gamma, alpha = 2.0, 0.25
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
#             (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

#     # 출처: https://3months.tistory.com/414 [Deep Play]
#
#     # def focal_losds(gamma=2., alpha=.25):
#     #     def focal_loss_fixed(y_true, y_pred):
#     #         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#     #         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#     #         return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - K.mean(
#     #             (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
#     #
#     #     return focal_loss_fixed
#
    adam = optimizers.Adam(lr= rate)  # 학습률
    # 컴파일 옵션 # loss='sparse_categorical_crossentropy',
    new_model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=adam,
                      metrics=['acc'])
    # ----------------------------------------another model --------------------
#     input_tensor = Input(shape=(150, 150, 3), dtype='float32', name='input')
#
#     pre_trained_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
#     pre_trained_vgg.trainable = False
#     pre_trained_vgg.summary()
#
#     model = models.Sequential()
#     model.add(pre_trained_vgg)
#     model.add(layers.Flatten())
#     model.add(layers.Dense(4096, activation='relu'))
    # model.add(Dropout(0.3))
#     model.add(layers.Dense(2048, activation='relu'))
    # model.add(Dropout(0.3))
#     model.add(layers.Dense(1024, activation='relu'))
    # model.add(Dropout(0.3))
#     model.add(layers.Dense(2, activation='softmax'))
#

#     model.summary()
#
#     adam = optimizers.Adam(lr=0.000001)  # 학습률
# #     # 컴파일 옵션 # loss='sparse_categorical_crossentropy',
#     model.compile(loss=[focal_loss],
#                         optimizer=adam,
#                         metrics=['acc'])
#

#
    return new_model


def my_metrics(y_true, y_pred):
    accuracy=accuracy_score(y_true, y_pred)
    precision=precision_score(y_true, y_pred,average='weighted')
    recall = recall_score(y_true, y_pred)
    # f1Score=f1_score(y_true, y_pred, average='weighted')
    f1Score = metrics.f1_score(y_true, y_pred, labels=np.unique(y_pred))
    print("Accuracy  : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("recall : {}".format(recall))
    print("f1Score : {}".format(f1Score))
    cm=confusion_matrix(y_true, y_pred)
    print(cm)
    return accuracy, precision, f1Score


def transfer_gan():
    g_path = "C:/Users/user/Desktop/deeplearning/gan/"   # gan 의 경로
    d_path = "C:/Users/user/Desktop/deeplearning/vgg16/train/defect/"  # train_defect 경로
    G_Files = os.listdir(g_path)


    if (len(G_Files) != 0):  # 파일이 0개가 아니면 실행되는 코드
        for i in G_Files:
            shutil.move(g_path + i, d_path)
    else:
        print("No file moved. Source empty!")


def back_gan():  # train에 있는 gan

    g_path = "C:/Users/user/Desktop/deeplearning/gan/"
    g1_path = "C:/Users/user/Desktop/deeplearning/gan_copy/"  # gan 의 경로
    d_path = "C:/Users/user/Desktop/deeplearning/vgg16/train/defect/"# train_defect 경로
    G_Files1 = os.listdir(g1_path)

    if (len(G_Files1) != 0):
        for j in G_Files1:
            shutil.move(d_path + j, g_path)

    else:
        print("No file moved. Source empty!")

# input image dimensions
img_rows, img_cols =  150, 150
train_path=datasetFolderName+'/train/'
validation_path=datasetFolderName+'/validation/'
test_path=datasetFolderName+'/test/'
Model=getModel()

# ===============Stratified K-Fold======================
skf = StratifiedKFold(n_splits=4 , shuffle=True, random_state = 42)
skf.get_n_splits(X, Y)
foldNum = 0
global arr
global arr1
arr = []
arr1 = []
for train_index, val_index in skf.split(X, Y):
    # First cut all images from validation to train (if any exists)
    transferAllClassBetweenFolders('validation', 'train', 1.0)
    foldNum += 1
    print("Results for fold", foldNum)
    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]

    # Move validation images of this fold from train folder to the validation folder
    for eachIndex in range(len(X_val)):
        classLabel = ''
        if (Y_val[eachIndex] == 0):
            classLabel = classLabels[0]

        #         elif(Y_val[eachIndex]==1):
        #             classLabel=classLabels[1]
        else:
            classLabel = classLabels[1]

            # classLabel=classLabels[2]
        # Then, copy the validation images to the validation folder
        shutil.move(datasetFolderName + '/train/' + classLabel + '/' + X_val[eachIndex],
                    datasetFolderName + '/validation/' + classLabel + '/' + X_val[eachIndex])


    # reduce learning rate for val_accuracy



    train_datagen = ImageDataGenerator(rescale=1. / 255)



    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Start ImageClassification Model
    transfer_gan()

    train_generator = train_datagen.flow_from_directory(
        train_path,
        shuffle=True,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='binary',
        ) # subset='training'

    validation_generator = validation_datagen.flow_from_directory(
        validation_path,
        shuffle=True,
        target_size = (img_rows, img_cols),
        batch_size = batch_size,
        class_mode = 'binary'  # only data, no labels
    )


    filename = 'checkpoint-epoch-{}-batch-{}-trial-{}.h5'.format(epoch, batch_size, foldNum)

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=30),
                 keras.callbacks.ModelCheckpoint(filepath= filename,
                                                 monitor='val_acc',
                                                 save_best_only=True)]
    # checkpoint = ModelCheckpoint(filename,  # file명을 지정합니다
    #                              monitor='val_loss',  # val_loss 값이 개선되었을때 호출됩니다
    #                              verbose=1,  # 로그를 출력합니다
    #                              save_best_only=True,  # 가장 best 값만 저장합니다
    #                              mode='min'  # auto는 알아서 best를 찾습니다. min/max
    #                              )
    #
    # earlystopping = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=15)


    history = Model.fit_generator(train_generator,
                                  epochs=epoch, validation_data=validation_generator, callbacks=callbacks) # callbacks=[checkpoint, earlystopping]

    foldNum = str(foldNum)
    drop = str(drop)
    rate = str(rate)

    name = "Vgg0608" + foldNum +"_" + drop +"_" + rate +".h5"
    Model.save(name)

    foldNum = int(foldNum)
    drop = float(drop)
    rate = float(rate)

    # =============predict============= 이게 원래 예측 코드 잠깐 꺼놈
    # predictions = Model.predict_generator(validation_generator, verbose=1)
    # yPredictions = np.argmax(predictions, axis=1)
    # true_classes = validation_generator.classes
    # # evaluate validation performance
    # print("***Performance on Validation data***")
    # valAcc, valPrec, valFScore = my_metrics(true_classes, yPredictions)

    # 시작

    model = load_model(name)
    validation_generator.reset()


    def my_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        print('오차행렬')
        print(cm)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred)
        # f1Score=f1_score(y_true, y_pred, average='weighted')
        f1Score = metrics.f1_score(y_true, y_pred, labels=np.unique(y_pred))
        print("Accuracy  : {}".format(accuracy))
        print("Precision : {}".format(precision))
        print("recall : {}".format(recall))
        print("f1Score : {}".format(f1Score))

        return accuracy, precision, recall, f1Score


    predictions = model.predict_generator(validation_generator, verbose=1)
    # print(predictions)

    print('=======true value==============')
    true_classes = validation_generator.classes
    # print(true_classes)

    # 이거는 하나씩 돌리는 코드
    # threshold 조절
    # predict_proba() 결과 값의 두 번째 컬럼, 즉 Positive 클래스의 컬럼 하나만 추출하여 Binarizer를 적용
    # custom_threshold = 0.3 # 1이될 확률이 0.3보다크면 1인걸로하래
    # predictions_thres = predictions[:,1].reshape(-1,1)
    # binarizer = Binarizer(threshold=custom_threshold).fit(predictions_thres)
    # custom_predict = binarizer.transform(predictions_thres)

    # 임계값
    thresholds = [0.1, 0.13, 0.15, 0.17, 0.2, 0.3, 0.4, 0.49, 0.4908, 0.491, 0.492, 0.495, 0.4995, 0.4999, 0.5, 0.55,
                  0.6, 0.65, 0.7, 0.75, 0.8, 0.9]


    # 평가지표를 조사하기 위한 새로운 함수 생성
    def get_eval_by_threshold(y_true, predictions_thres, thresholds):
        # thresholds list 객체 내의 값을 iteration 하면서 평가 수행
        for custom_threshold in thresholds:
            binarizer = Binarizer(threshold=custom_threshold).fit(predictions_thres)
            custom_predict = binarizer.transform(predictions_thres)
            print('\n임계값: ', custom_threshold)
            my_metrics(y_true, custom_predict)


    get_eval_by_threshold(true_classes, predictions[:, 1].reshape(-1, 1), thresholds)

    validation_Ok = "C:/Users/user/Desktop/deeplearning/vgg16/validation/Ok/"
    validation_defect = "C:/Users/user/Desktop/deeplearning/vgg16/validation/defect/"

    v_Ok = os.listdir(validation_Ok)
    # v_Ok = np.array(v_Ok)
    v_defect = os.listdir(validation_defect)
    # v_defect = np.array(v_defect)

    arr.append(v_Ok)
    arr1.append(v_defect)

    print(arr)
    print(arr1)
    # train_defect에 있는 gan 폴더로 다시 옮기기
    back_gan()
    #
    # # ================code that i made
    # Y_pred = model.predict_generator(validation_generator,  verbose=1 ) # num_of_test_samples // batch_size + 1,
    # classes = validation_generator.classes[validation_generator.index_array]
    # y_pred = np.argmax(Y_pred, axis=1)
    # print('Confusion Matrix')
    # print(confusion_matrix(validation_generator.classes, y_pred))
    # print('Classification Report')
    # target_names = ['Ok', 'defect']
    # print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

       # 모델 저장하기
    # 최종 결과 리포트
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    #
    # epochs = range(len(acc))
    #
    #
    #
    # plt.plot(epochs, acc, 'r', label='Training acc')
    # plt.plot(epochs, val_acc, 'b', label='validation acc')
    # plt.title('Training and validation accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('acc')
    # plt.ylim([0.4, 1])
    # plt.legend()
    # plt.figure()
    #
    # plt.plot(epochs, loss, 'r', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    #
    # plt.legend()
    # plt.show()


test_Ok = "C:/Users/user/Desktop/deeplearning/vgg16/test/Ok/"
test_defect = "C:/Users/user/Desktop/deeplearning/vgg16/test/defect/"

test_Ofiles = os.listdir(test_Ok)
test_dfiles = os.listdir(test_defect)

print("validation data")
print(arr)
print("-------------------")
print(arr1)



print("test data")
print(test_Ofiles)
print("-------------------")
print(test_dfiles)



# def validation():
#     return arr, arr1
#
#
# validation()

#------------------don't run ===================
# # 저장 모델 불러오기
#
# new_model = load_model("Vgg0426.h5")
#====================================
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# ​
# plt.figure()
# plt.plot(loss, 'ro-')
# plt.plot(val_loss, 'bo-')
# plt.ylabel('Cross Entropy')
# plt.xlabel('Epoch')
# plt.title('Training and Validation Loss')
# plt.show()
#  https://m.blog.naver.com/jeonghj66/222004874975