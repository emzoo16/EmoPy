import os
import sys
import cv2

CROPPED_CK_DIRECTORY = "../examples/image_data/Cropped-CK+"
ORIGINAL_CK_DIRECTORY = "../examples/image_data/CK+"
HAARCASCADE_PATH = "../examples/haarcascade_frontalface_default.xml"


def crop_and_save_face(image_file, emotion_label):
    face_detector = cv2.CascadeClassifier(HAARCASCADE_PATH)

    print("cropping " + image_file)

    image = cv2.imread(ORIGINAL_CK_DIRECTORY + "/" +
                       emotion_label + "/" + image_file)
    faces = face_detector.detectMultiScale(image, 1.3, 5)
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48),
                          interpolation=cv2.INTER_AREA)

        cv2.imwrite(CROPPED_CK_DIRECTORY + "/" + emotion_label +
                    "/" + image_file, face)


def create_cropped_image_directory():
    os.mkdir(CROPPED_CK_DIRECTORY)

    if os.path.exists(ORIGINAL_CK_DIRECTORY):
        label_directories = [dir for dir in os.listdir(
            ORIGINAL_CK_DIRECTORY) if not dir.startswith('.')]
        for label_directory in label_directories:
            os.mkdir(CROPPED_CK_DIRECTORY + "/" + label_directory)

            print("processing " + label_directory + "...")
            image_files = [image_file for image_file in os.listdir(
                ORIGINAL_CK_DIRECTORY + "/" + label_directory) if not image_file.startswith('.')]
            for image_file in image_files:
                crop_and_save_face(image_file,
                                   label_directory)
    print("done!")


if os.path.exists(CROPPED_CK_DIRECTORY):
    print("Cropped dataset already exists\n")
    exit()

create_cropped_image_directory()
