# utf-8

"""
train_model.py

Trains a facial recognition model using custom images of celebrities and saves the face encodings and 
names in a pickle file for transfer learning.

Consists of:
    custom_faces_folder: The path containing subfolders of custom images. Each directory names the person of its files.
    output_encodings_file: Output pkl file containing the encodings and names.
"""

import face_recognition
import os
import pickle


def train_model(custom_faces_folder, output_encodings_file='custom_encodings.pkl'):
    known_face_encodings = []
    known_face_names = []

    # Training with custom images
    print(f"Custom images: {custom_faces_folder}")
    for root, dirs, files in os.walk(custom_faces_folder):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                path = os.path.join(root, file)
                print(f"{path}")
                image = face_recognition.load_image_file(path)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(os.path.basename(root))
                    print(f"Added encoding: {os.path.basename(root)}")

    # Verify that we have encodings before saving
    if known_face_encodings and known_face_names:
        print(f"Saving encodings: {output_encodings_file}")
        data = {"encodings": known_face_encodings, "names": known_face_names}
        with open(output_encodings_file, 'wb') as f:
            pickle.dump(data, f)
        print("Training completed")
    else:
        print("No encodings found to save")


if __name__ == "__main__":
    # Path to custom images
    custom_faces_folder = 'data/Hollywood_Celebrity_Facial_Recognition_Dataset'
    train_model(custom_faces_folder)
