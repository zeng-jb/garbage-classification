import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from keras.models import Sequential
from keras.models import load_model
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, os, random

# 数据集路径base_path
base_path = './dataset-resized'
img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))
print(len(img_list))

# Batch Size定义：一次训练所选取的样本数。
batch_size_value = 16

"""
数据处理
:数据集路径base_path
:train, test:处理后的训练集数据、测试集数据
"""
train_datagen = ImageDataGenerator(
    # 对图片的每个像素值均乘上这个放缩因子，把像素值放缩到0和1之间有利于模型的收敛
    rescale=1. / 225,
    # 浮点数，剪切强度（逆时针方向的剪切变换角度）
    shear_range=0.1,
    # 随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
    zoom_range=0.1,
    # 浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
    width_shift_range=0.1,
    # 浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
    height_shift_range=0.1,
    # 布尔值，进行随机水平翻转
    horizontal_flip=True,
    # 布尔值，进行随机竖直翻转
    vertical_flip=True,
    # 在 0 和 1 之间浮动。用作验证集的训练数据的比例
    validation_split=0.1

)

# 接下来生成测试集，可以参考训练集的写法
test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.1

)

train_generator = train_datagen.flow_from_directory(
    # 提供的路径下面需要有子目录
    base_path,
    # 整数元组 (height, width)，默认：(256, 256)。 所有的图像将被调整到的尺寸。
    target_size=(300, 300),
    # 一批数据的大小已经定义16个
    batch_size=batch_size_value,
    # "categorical", "binary", "sparse", "input" 或 None 之一。
    # 默认："categorical",返回one-hot 编码标签。
    class_mode='categorical',
    # 数据子集 ("training" 或 "validation")
    subset='training',
    seed=0

)

validation_generator = test_datagen.flow_from_directory(
    base_path,
    target_size=(300, 300),
    batch_size=batch_size_value,
    class_mode='categorical',
    subset='validation',
    seed=0

)

labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
print(labels)

# 模型建立
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


# 绘制损失函数图像
class LossHistory(keras.callbacks.Callback):
    # 函数开始时创建盛放loss与acc的容器
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    # 按照batch来进行追加数据
    def on_batch_end(self, batch, logs={}):
        # 每一个batch完成后向容器里面追加loss，acc
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        # 每一个epoch完成后向容器里面追加loss，acc
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        # plt.show()
        plt.savefig('./results/learning_rate.png')


history = LossHistory()

# 编译模型, 采用 compile 函数: https://keras.io/models/model/#compile
# optimizer是优化器, 主要有Adam、sgd、rmsprop等方式。
# loss损失函数,多分类采用 categorical_crossentropy
# metrics是除了损失函数值之外的特定指标, 分类问题一般都是准确率
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# 使用 Python 生成器（或 Sequence 实例）逐批生成的数据，按批次训练模型。model.fit_generator
# 一个生成器或 Sequence 对象的实例 train_generator,
# epochs: 整数，数据的迭代总轮数。epochs=100,
# 一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。steps_per_epoch=2276//32,
# 验证集 validation_data=validation_generator,
# 在验证集上,一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。validation_steps=251//32,
model.fit_generator(train_generator, epochs=100, steps_per_epoch=2276 // 32, validation_data=validation_generator,
                    validation_steps=251 // 32, callbacks=[history])

history.loss_plot('epoch')

# 保存模型
model.save('./results/my_model_test1.h5')
