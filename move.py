import random
import os
import shutil
import time

# kfold로 분배된 파일을 다시 train으로 옮겨주는 코드


def move():
    train_Ok = "C:/Users/user/Desktop/deeplearning/vgg16/train/Ok/"
    train_defect = "C:/Users/user/Desktop/deeplearning/vgg16/train/defect/"

    validation_Ok = "C:/Users/user/Desktop/deeplearning/vgg16/validation/Ok/"
    validation_defect = "C:/Users/user/Desktop/deeplearning/vgg16/validation/defect/"

    test_Ok = "C:/Users/user/Desktop/deeplearning/vgg16/test/Ok/"
    test_defect = "C:/Users/user/Desktop/deeplearning/vgg16/test/defect/"

    validation_Ofiles = os.listdir(validation_Ok)
    validation_dfiles = os.listdir(validation_defect)

    test_Ofiles = os.listdir(test_Ok)
    test_dfiles = os.listdir(test_defect)

    for a in validation_Ofiles:
        shutil.move(validation_Ok + a, train_Ok + a)

    for b in test_Ofiles:
        shutil.move(test_Ok + b, train_Ok + b)

    for c in validation_dfiles:
        shutil.move(validation_defect + c, train_defect + c)

    for d in test_dfiles:
        shutil.move(test_defect + d, train_defect + d)


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


move()

train_Ok = "C:/Users/user/Desktop/deeplearning/vgg16/train/Ok/"
train_defect = "C:/Users/user/Desktop/deeplearning/vgg16/train/defect/"

ok_files = os.listdir(train_Ok)
defect_files = os.listdir(train_defect)

print("Ok :", len(ok_files))
print("defect :", len(defect_files))


back_gan()


