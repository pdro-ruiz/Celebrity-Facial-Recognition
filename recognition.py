# utf-8

"""
recognition.py

Performs facial recognition on a video using previously saved face encodings.
Detects and labels faces in each frame of the video, marking unknown faces as "unknown".

Consists of:
    encodings_path: The path to the pickle file.
    video_path: The path to the video.
"""

import cv2
import face_recognition
import pickle


encodings_path = 'custom_encodings.pkl'
video_path = 'data/sample.mp4'

# Load the encodings
with open(encodings_path, 'rb') as f:
    data = pickle.load(f)

# Start the video
print("Starting facial recognition in video...")
video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print("Could not open the video.")
else:
    print("Video opened successfully.")
    frame_count = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            print(f"error frame {frame_count + 1}")
            break

        frame_count += 1

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Recognizing faces in the frame and extracting their encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # start searching for the encoding and comparing it with our data, mark as unknown if it is not found.
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Comparar las caras detectadas con nuestros encodings
            matches = face_recognition.compare_faces(data["encodings"], face_encoding)
            name = "unknown"

            # use of the face_distance to obtain the closest similarity to 0
            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = data["names"][best_match_index]

            # paint the rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Paint the face label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Show the result
        cv2.imshow('Video', frame)

        # Exit with the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("press q to exit")
            break

    # Release the video pointer and close the window.
    video_capture.release()
    cv2.destroyAllWindows()




