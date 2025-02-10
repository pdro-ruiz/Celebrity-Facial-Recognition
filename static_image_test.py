# utf-8

"""
static_image_test.py

Loads an image from the database trained in custom_encodings.pkl to perform facial recognition.

Consists of:
    image_path: The path to the image.
    encodings_path: The path to the pickle file.
"""


import face_recognition
import cv2
import matplotlib.pyplot as plt
import pickle


image_path = 'data/Hollywood_Celebrity_Facial_Recognition_Dataset/Jennifer Lawrence/066_4c979163.jpg'
encodings_path = 'custom_encodings.pkl'

image = face_recognition.load_image_file(image_path)
face_encodings = face_recognition.face_encodings(image)

# Load the encodings
with open(encodings_path, 'rb') as f:
    data = pickle.load(f)

# Convert the image to RGB
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Perform facial recognition
face_locations = face_recognition.face_locations(image)

# Iterate over the detected faces and compare them with our encodings
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(data['encodings'], face_encoding)
    name = "Unknown"                                                                            # If no match is found, the name is "Unknown"

    if True in matches:
        first_match_index = matches.index(True)
        name = data['names'][first_match_index]

    # Draw the box around the face
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

    # Label with the name
    font = cv2.FONT_HERSHEY_DUPLEX
    text = name
    text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
    text_x = left
    text_y = top - 10 if top - 10 > 10 else top + 10
    cv2.rectangle(image, (left, top - text_size[1] - 10), (left + text_size[0], top), (0, 0, 255), cv2.FILLED)
    cv2.putText(image, text, (text_x, text_y), font, 0.5, (255, 255, 255), 1)

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
