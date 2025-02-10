# Celebrity-Facial-Recognition

This project is an introductory facial recognition system designed to identify celebrities in video footage. It serves as an example of the potential of facial recognition technology, demonstrating that even with limited data and resources, it is possible to implement a viable face detection system.

**Features:**
- Load and process video files to detect faces.
- Identify known celebrities by comparing against saved face encodings.
- Label recognized faces with their names, and mark unrecognized faces as "Unknown".
- Visualize the results by drawing rectangles and labels around detected faces in video frames.

**Usage:**
1. Train the model with a dataset of celebrity images to create a `custom_encodings.pkl` file.
2. Use the recognition script to process video files and identify fac

**Limitations and Future Work:**
This initial implementation has several limitations:
- Performance issues, as it can be improved significantly.
- Difficulties in identifying faces at different angles, especially profiles or other non-frontal views.
- Challenges in recognizing faces with accessories like glasses.
Despite these limitations, the project shows that with minimal data and capabilities, a functional face detection system can be implemented. Future work will focus on refining these features and introducing new improvements.

**Dependencies:**
- Python 3.x
- `face_recognition` library
- OpenCV
- Matplotlib
- Pickle

**Contributing:**
Contributions are welcome! Please fork the repository and submit a pull request.

