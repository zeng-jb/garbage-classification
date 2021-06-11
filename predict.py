import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from tensorflow.keras.models import Sequential
from keras.models import Sequential
import glob, os, random

from tensorflow.python.keras.models import load_model


def predict(img):
    """
    加载模型和模型预测
    主要步骤:
        1.加载模型(请加载你认为的最佳模型)
        2.图片处理
        3.用加载的模型预测图片的类别
    :param img: PIL.Image 对象
    :return: string, 模型识别图片的类别,
            共 'cardboard','glass','metal','paper','plastic','trash' 6 个类别
    """

    # 把图片转换成为numpy数组
    img = img.resize((300, 300))
    imgs = image.img_to_array(img)

    # 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
    # 如果你的模型是在 results 文件夹下的 dnn.h5 模型，则 model_path = 'results/my_model.h5'
    model_path = 'results/my_model.h5'
    '''
    try:
        # 作业提交时测试用, 请勿删除此部分
        model_path = os.path.realpath(__file__).replace('main.py', model_path)
    except NameError:
        model_path = './' + model_path
    '''

    # -------------------------- 实现模型预测部分的代码 ---------------------------
    # 加载模型
    model = load_model(model_path)

    # expand_dims的作用是把img.shape转换成(1, img.shape[0], img.shape[1], img.shape[2])
    x = np.expand_dims(imgs, axis=0)

    # 模型预测
    y = model.predict(x)

    # 获取labels
    labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

    # -------------------------------------------------------------------------
    predict = labels[np.argmax(y)]

    plt.figure()
    plt.title('pred:%s ' % (predict))
    plt.imshow(img)
    plt.savefig('./results/result_test.png')

# 返回图片的类别
    return predict

#读取图片并显示
from PIL import Image, ImageDraw
#指定路径
sample_image_path = os.path.join( 'testImg/paper1.jpg')
#读入图片
sample_image = Image.open(sample_image_path)
print ( sample_image.format, "%dx%d" % sample_image.size, sample_image.mode)
#输出
# plt.title('Sample Image')
# plt.imshow(sample_image)
# plt.show()
print(predict(sample_image))