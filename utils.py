import cv2
import numpy as np


def process_image(img_path):
    """
    Function that preprocessing the image for modelling or predicting later
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.shape)
    # Crop image
    rows = (img != 0).any(axis=1)
    cols = (img != 0).any(axis=0)
    img = img[rows, :][:, cols]

    # Padding with 0s
    num_rows, num_cols = img.shape
    add_rows = num_rows // 6
    add_cols = num_cols // 6

    img = np.r_[np.zeros((add_rows, num_cols)), img, np.zeros((add_rows, num_cols))]
    new_rows = img.shape[0]
    img = np.c_[np.zeros((new_rows, add_cols)), img, np.zeros((new_rows, add_cols))]
    # cv2.imshow("img", img)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # img = cv2.flip(img, 1)
    # # Rotate the image 90 degrees to the left
    # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


    return img