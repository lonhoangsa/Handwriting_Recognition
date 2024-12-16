import os
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from model.KNN import KNN
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier



def preprocess_image(image_path: str):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("img", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Apply thresholding to binarize the image
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of the characters
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # Extract individual characters
    characters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 10 or h < 10:
            continue
        char_image = binary_image[y:y + h, x:x + w]
        # Calculate the coverage
        char_area = w * h
        total_area = char_image.shape[0] * char_image.shape[1]
        coverage = char_area / total_area

        char_image = cv2.flip(char_image, 1)
        # Rotate the image 90 degrees to the left
        char_image = cv2.rotate(char_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        char_image = cv2.resize(char_image, (28, 28))  # Resize to 28x28 pixels minus the margin
        char_image = cv2.copyMakeBorder(char_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        char_image = cv2.resize(char_image, (28, 28))
        characters.append(char_image)
        plt.imshow(char_image, cmap='gray')
        plt.show()

    return characters

def load_model(model_name = 'knn'):
    model_path ="model/" + model_name + ".pkl"

    def data_to_numpy(data):
        data_loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)
        images, labels = next(iter(data_loader))
        return images.view(len(data), -1).numpy(), labels.numpy()

    transform = transforms.Compose([transforms.ToTensor()])

    if model_name == 'knn':
        if os.path.exists(model_path):
            print("Loading pre-trained model...")
            model = KNN.load_model(model_path)
            print("Model loaded.")
        else:
            print("Training new model...")
            trainset_emnist = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True,
                                                          transform=transform)

            X_train, y_train = data_to_numpy(trainset_emnist)

            model = KNN(K=5)
            model.fit(X_train, y_train)
            model.save_model(model_path)
            print("Model trained and saved.")

    elif model_name == 'svm':
        if os.path.exists(model_path):
            # Load the model from the file
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            print("Model loaded from file.")
        else:
            print("Training new model...")
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
        if os.path.exists(model_path):
            # Load the model from the file
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            print("Model loaded from file.")
        else:
            print("Training new model...")
            # Load and preprocess data
            trainset_emnist = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True,
                                                          transform=transform)

            X_train, y_train = data_to_numpy(trainset_emnist)

            model = RandomForestClassifier(n_estimators=500)
            model.fit(X_train, y_train)
            print("Model trained.")

            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
            print("Model saved to file.")

    return model

def predict_characters(characters, model):
    characters = np.array(characters).reshape(len(characters), -1)  # Flatten the images
    predictions = model.predict(characters)
    return predictions
