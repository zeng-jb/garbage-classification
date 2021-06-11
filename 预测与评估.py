###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from keras.models import Sequential
import glob, os, random

base_path = './dataset-resized'
img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))
print(len(img_list))
for i, img_path in enumerate(random.sample(img_list, 6)):
    img = load_img(img_path)
    img = img_to_array(img, dtype=np.uint8)

    plt.subplot(2, 3, i + 1)
    plt.imshow(img.squeeze())
train_datagen = ImageDataGenerator(
    rescale=1. / 225, shear_range=0.1, zoom_range=0.1,
    width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,
    vertical_flip=True, validation_split=0.1)

test_datagen = ImageDataGenerator(
    rescale=1. / 255, validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
    base_path, target_size=(300, 300), batch_size=16,
    class_mode='categorical', subset='training', seed=0)

validation_generator = test_datagen.flow_from_directory(
    base_path, target_size=(300, 300), batch_size=16,
    class_mode='categorical', subset='validation', seed=0)

labels = (train_generator.class_indices)
print(labels)
labels = dict((v, k) for k, v in labels.items())

print(labels)
from keras.models import load_model

# del model
# 加载模型
model = load_model('./results/my_model_test1.h5')
# 获取验证集的 loss 和 accuracy
loss, accuracy = model.evaluate_generator(validation_generator)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))
test_x, test_y = validation_generator.__getitem__(8)

preds = model.predict(test_x)

plt.figure(figsize=(20, 12))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.title('pred:%s / truth:%s' % (labels[np.argmax(preds[i])], labels[np.argmax(test_y[i])]))
    plt.imshow(test_x[i])
plt.savefig('./results/result.png')

