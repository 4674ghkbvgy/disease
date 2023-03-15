import numpy as np
from sklearn import svm
import joblib
# 加载数据
train_data = np.load('/home/zty/project/disease/dataset_b/train.npy', allow_pickle=True).item()
test_data = np.load('/home/zty/project/disease/dataset_b/test.npy', allow_pickle=True).item()

# 将图像数据的像素值范围缩放到 [0, 1] 之间
train_images = train_data['images'] / 255.0
test_images = test_data['images'] / 255.0

# 将图像数据转换为一维向量
train_vectors = train_images.reshape(train_images.shape[0], -1)
test_vectors = test_images.reshape(test_images.shape[0], -1)

# 定义模型
model = svm.SVC(kernel='linear', C=1, decision_function_shape='ovr')

# 训练模型
model.fit(train_vectors, train_data['labels'])

# 评估模型
test_predictions = model.predict(test_vectors)
test_accuracy = np.mean(test_predictions == test_data['labels'])
print('Test accuracy:', test_accuracy)
joblib.dump(model, 'svm_model_b.pkl')