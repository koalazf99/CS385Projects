import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix

import torchvision.datasets as datasets
import torchvision.transforms as transforms

train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)
# convert to numpy data
train_data = train_dataset.train_data.numpy()
train_label = train_dataset.train_labels.numpy()
test_data = test_dataset.test_data.numpy()
test_label = test_dataset.test_labels.numpy()

# Normalization
train_data = train_data / 255.0
test_data = test_data / 255.0

train_data = train_data.reshape(train_data.shape[0], -1)
test_data = test_data.reshape(test_data.shape[0], -1)

# Training SVM using sklearn
model = svm.SVC(C=5, gamma=0.05, max_iter=20)
model.fit(train_data, train_label)

# Training result
train_result = model.predict(train_data)
accuracy = np.sum(train_result == train_label) / train_label.shape[0]
print("Training Accuracy of the model on the {} train images: {} %"
      .format(train_label.shape[0], accuracy))

# Testing result
test_result = model.predict(test_data)
accuracy = np.sum(test_result == test_label) / test_label.shape[0]
print("Testing Accuracy of the model on the {} test images: {} %"
      .format(test_label.shape[0], accuracy))

# Temp Result Confusion Matrix
matrix = confusion_matrix(test_label, test_result)
