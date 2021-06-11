# 上海开始施行垃圾分类啦。那么我们能不能通过平常学习的机器学习和深度学习的算法来实现一个简单的垃圾分类的模型呢？
# 下面主要用过CNN来实现垃圾的分类。在本数据集中，垃圾的种类有六种（和上海的标准不一样），分为玻璃、纸、硬纸板、塑料、金属、一般垃圾。
# 该数据集包含了2527个生活垃圾图片。数据集的创建者将垃圾分为了6个类别，分别是：

# 玻璃（glass）共501个图片
# 纸（paper）共594个图片
# 硬纸板（cardboard）共403个图片
# 塑料（plastic）共482个图片
# 金属（metal）共410个图片
# 一般垃圾（trash）共137个图片
# 物品都是放在白板上在日光/室内光源下拍摄的，压缩后的尺寸为512 * 384。

# dataset from https://github.com/garythung/trashnet/tree/master/data
# Unzip data/dataset-resized.zip

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from tensorflow.keras.models import Sequential
from keras.models import Sequential
import glob, os, random

from tensorflow.python.keras.models import load_model

base_path ='./dataset-resized'
img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))
print(len(img_list))

# 我们总共有2527张图片。我们随机展示其中的6张图片
for i, img_path in enumerate(random.sample(img_list, 6)):
    img = load_img(img_path)
    img = img_to_array(img, dtype=np.uint8)
    
    plt.subplot(2, 3, i+1)
    plt.imshow(img.squeeze())
#对数据进行分组
#ImageDataGenerator()是keras.preprocessing.image模块中的图片生成器，可以每一次给模型“喂”一个batch_size大小的样本数据，
#同时也可以在每一个批次中对这batch_size个样本数据进行增强，扩充数据集大小，增强模型的泛化能力。比如进行旋转，变形，归一化等等。

train_datagen = ImageDataGenerator(
    rescale=1./225, shear_range=0.1, zoom_range=0.1,
    width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,
    vertical_flip=True, validation_split=0.1)
#shear_range 剪切强度（逆时针方向的剪切变换角度)
#validation_split: 保留用于验证的图像的比例（严格在0和1之间）
test_datagen = ImageDataGenerator(
    rescale=1./255, validation_split=0.1)
    
train_generator = train_datagen.flow_from_directory(
    base_path, target_size=(300, 300), batch_size=16,
    class_mode='categorical', subset='training', seed=0)

validation_generator = test_datagen.flow_from_directory(
    base_path, target_size=(300, 300), batch_size=16,
    class_mode='categorical', subset='validation', seed=0)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

print(labels)
model = Sequential([
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D(pool_size=2),

    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Flatten(),

    Dense(64, activation='relu'),

    Dense(6, activation='softmax')
])
#(交叉熵损失函数) 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit_generator(train_generator, epochs=100, steps_per_epoch=2276//32,validation_data=validation_generator,
                    validation_steps=251//32)

#参数steps_per_epoch是通过把训练图像的数量除以批次大小得出的。例如,有100张图像且批次大小为50,则steps_per_epoch值为2
#结果展示 下面我们随机抽取validation中的16张图片，展示图片以及其标签，并且给予我们的预测。 我们发现预测的准确度还是蛮高的，对于大部分图片，都能识别出其类别。
test_x, test_y = validation_generator.__getitem__(1)

preds = model.predict(test_x)

plt.figure(figsize=(16, 16))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.title('pred:%s / truth:%s' % (labels[np.argmax(preds[i])], labels[np.argmax(test_y[i])]))
    plt.imshow(test_x[i])



