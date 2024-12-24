import os
import pickle
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2 import Mat
from numpy import ndarray
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from models.KNN import KNN
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier



def preprocess_image(image_path) -> list[Mat | ndarray | ndarray]:
    # Apply thresholding to binarize the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plt.imshow(image)
    # plt.title('Original Image')
    # plt.show()
    copy_image = image.copy()
    # Invert the image to have black letters on white background
    image = cv2.bitwise_not(image)
    
    ret, thresh1 = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    # Sort contours from left to right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # Extract individual characters
    characters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        print(x,y,w,h)
        if w < 10 or h < 10:
            continue

        char_image = copy_image[y:y + h, x:x + w]

        num_rows, num_cols = char_image.shape
        add_rows = num_rows // 6
        add_cols = num_cols // 6
        char_image = cv2.flip(char_image, 1)
        # Rotate the image 90 degrees to the left
        char_image = cv2.rotate(char_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        char_image = cv2.copyMakeBorder(char_image, add_rows, add_rows, add_cols, add_cols, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        char_image = cv2.resize(char_image, (28, 28), interpolation=cv2.INTER_AREA)

        plt.imshow(char_image)
        plt.show()

        characters.append(char_image)

    return characters

def load_model(model_name = 'knn'):
    model_path =r"models/" + model_name + ".pkl"

    def data_to_numpy(data):
        data_loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)
        images, labels = next(iter(data_loader))
        return images.view(len(data), -1).numpy(), labels.numpy()

    transform = transforms.Compose([transforms.ToTensor()])

    if model_name == 'knn':
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            print("Model loaded from file.")
        else:
            print("Training new models...")
            trainset_emnist = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True,
                                                          transform=transform)

            X_train, y_train = data_to_numpy(trainset_emnist)

            model = KNN(k=3, weights='distance', algorithm='brute', p=2)
            model.fit(X_train, y_train)

            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
            print("Model saved to file.")

    elif model_name == 'svm':
        model_path = r"models/" + model_name + ".pkl"
        if os.path.exists(model_path):
            # Load the models from the file
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            print("Model loaded from file.")
        else:
            print("Training new models...")
            # Load and preprocess data
            trainset_emnist = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True,
                                                          transform=transform)
            X_train, y_train = data_to_numpy(trainset_emnist)
            model = SVC(kernel='rbf')
            model.fit(X_train, y_train)
            print("Model trained.")

            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
            print("Model saved to file.")
    elif model_name == 'rf':
        model_path = r"models/" + model_name + ".pkl"
        if os.path.exists(model_path):
            # Load the models from the file
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            print("Model loaded from file.")
        else:
            print("Training new models...")
            # Load and preprocess data
            trainset_emnist = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True,
                                                          transform=transform)
            X_train, y_train = data_to_numpy(trainset_emnist)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            print("Model trained.")

            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
            print("Model saved to file.")
    elif model_name == 'lr':
        model_path = r"models/" + model_name + ".pkl"
        if os.path.exists(model_path):
            # Load the models from the file
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            print("Model loaded from file.")
        else:
            print("Training new models...")
            # Load and preprocess data
            trainset_emnist = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True,
                                                          transform=transform)
            X_train, y_train = data_to_numpy(trainset_emnist)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            print("Model trained.")

            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
            print("Model saved to file.")
    else:
        raise ValueError(f"Unsupported models name: {model_name}")
    return model


def predict_characters(characters, model):
    characters = np.array(characters).reshape(len(characters), -1)  # Flatten the images
    predictions = model.predict(characters)
    return predictions

def retrain_model(model_name='knn', **params):
    model_path = "models/" + model_name + ".pkl"
    transform = transforms.Compose([transforms.ToTensor()])
    def data_to_numpy(data):
        data_loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)
        images, labels = next(iter(data_loader))
        return images.view(len(data), -1).numpy(), labels.numpy()
    print("Retraining models...")

    trainset_emnist = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
    X_train, y_train = data_to_numpy(trainset_emnist)

    if model_name == 'knn':
        model = KNN(**params)
        model.fit(X_train, y_train)
    elif model_name == 'svm':
        model = SVC(**params)
        model.fit(X_train, y_train)
    elif model_name == 'rf':
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unsupported models name: {model_name}")

    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print("Model retrained and saved.")