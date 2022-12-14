from typing import List
import imutils
import numpy as np
import cv2
import easyocr


def generate_ocr(gpu=True, model_path=r"E:\NoBackup\DGMD_E-14_FinalProject\ocr\models"):
    return easyocr.Reader(["en"], gpu=gpu, model_storage_directory=model_path)


def _create_image(image_file: str) -> np.ndarray:
    image = cv2.imread(image_file)
    # No need to convert back to RGB, easyocr can handle BGR
    return image


def _sharpen_image(image) -> np.ndarray:
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(image, -1, kernel)
    return image_sharp


def _generate_single_prediction_set(reader: easyocr.Reader, image: np.ndarray) -> List[str]:
    # By default, easyocr outputs list format with each element being a tuple of
    # ([bounding box 2D array], text, confidence)
    full_prediction = reader.readtext(image)
    predictions = []
    for word in full_prediction:
        conf = word[2]
        text = word[1]
        if conf > 0 and len(text) > 0:
            predictions.append(text)
    return predictions


def generate_predictions(reader, image_file: str, rotate=False) -> List[List[str]]:
    """
    Generates predictions from the specified image file, optionally rotating it by 90, 180, and 270 degrees (useful for
    testing accuracy against test images).

    Note that the keras-ocr pipeline object is assumed to be created by the caller.  This is to facilitate batch
    testing, so the pipeline doesn't get reinitialized for each image tested.

    Note that the returned object will NOT be an empty List if no text is recognized.  It will instead contain one or
    more empty Lists, each from a permutation of the supplied image.

    :param reader: easyocr Reader, created/initialized outside this method so that batch testing doesn't have to
                   recreate it for each image tested
    :param image_file: path of the image file to check for text
    :param rotate: True if predictions should also be generated for the image at 90, 180, and 270 degrees, intended
                   for use with batch testing of stock images that may be rotated, defaults to False
    :return: List containing prediction groups for each image permutation, where each prediction group is itself a
             List of text strings.  Note that if no text is found, this will NOT be an empty List, but will instead
             contain multiple empty Lists
    """
    base_image = _create_image(image_file)
    image_sharp = _sharpen_image(base_image)
    images = [base_image, image_sharp]
    all_predictions = []
    for image in images:
        # Rotate the image across 0, 90, 180, and 270, in case the pill is rotated, to improve readability
        if rotate:
            for angle in [0, 90, 180, 270]:
                if angle == 0:
                    rotated = image
                else:
                    rotated = imutils.rotate_bound(image, angle)
                all_predictions.append(_generate_single_prediction_set(reader, rotated))
        else:
            all_predictions.append(_generate_single_prediction_set(reader, image))

    return all_predictions
