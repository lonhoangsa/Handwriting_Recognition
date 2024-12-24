import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import joblib
from tqdm import tqdm  # Import tqdm

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load EMNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
testset = torchvision.datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

# Convert EMNIST to NumPy
def emnist_to_numpy(data):
    data_loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)
    images, labels = next(iter(data_loader))
    return images.view(len(data), -1).numpy(), labels.numpy()

# Chuyển dữ liệu thành numpy
X_train, y_train = emnist_to_numpy(trainset)
X_test, y_test = emnist_to_numpy(testset)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Starting training...")

# Subset the test data
_, X_test_subset, _, y_test_subset = train_test_split(X_test, y_test, test_size=1000, stratify=y_test, random_state=42)

# Khởi tạo mô hình Logistic Regression
lr_model = LogisticRegression(max_iter=100, 
                              solver='saga',
                              penalty='elasticnet',
                              tol=0.0001, 
                              C = 0.3, 
                              l1_ratio=0.5,
                              #multi_class='multinomial', 
                              warm_start=False,
                              verbose=1
                              )

# Huấn luyện mô hình và theo dõi tiến độ
# Vì .fit() không hỗ trợ trực tiếp, ta sẽ dùng tqdm trong một vòng lặp giả lập (ở đây không có vòng lặp cho Logistic Regression)
# Cách khác là theo dõi thời gian huấn luyện
lr_model.fit(X_train, y_train)

# Predict and evaluate Logistic Regression model
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print(f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")

# Save the model to a file
joblib.dump(lr_model, 'lr_model.pkl')
