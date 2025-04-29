# SignLanguageActionDetection

![Sign Language Detection Example](path/to/your/image.jpg)  
*Example of landmark detection on a video frame using MediaPipe.*

This repository contains the code and resources for the **SignLanguageActionDetection** project, a machine learning-based system designed to detect and classify sign language actions from video input. The project aims to build an efficient model capable of recognizing gestures such as "hello," "thanks," and "iloveyou," providing a foundation for applications that facilitate communication for the hearing-impaired or enable sign language integration into interactive systems.

## Key Features
- **Real-time Video Processing**: Analyzes video frames to detect and classify sign language gestures in real time.
- **Landmark Detection**: Utilizes MediaPipe to extract keypoints from the face, pose, and hands.
- **Sequence Classification**: Employs an LSTM-based neural network to classify sequences of keypoints into predefined actions.
- **Data Collection**: Supports scalable collection of video sequences for training and testing.

## Technologies Used
- **OpenCV**: For video capture and image processing.
- **MediaPipe**: For detecting and extracting landmarks from video frames.
- **TensorFlow/Keras**: For building and training the LSTM model.
- **Google Colab**: The code is optimized to run in a Google Colab environment with GPU support.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/SignLanguageActionDetection.git
   ```
2. **Install Dependencies**:
   Install the required libraries in your environment:
   ```bash
   pip install mediapipe opencv-python-headless tensorflow
   ```
3. **Run in Google Colab**:
   - Upload the `SignLanguageActionDetection.ipynb` notebook to Google Colab.
   - Use a webcam or upload a video file (e.g., `WIN_20240422_10_56_53_Pro.mp4`) for processing.
   - Execute the notebook cells to collect data, train the model, and test the system.

## Data Collection and Preprocessing
- **Actions**: The system detects three actions: "hello," "thanks," and "iloveyou."
- **Sequences**: For each action, 30 video sequences are collected, each consisting of 30 frames.
- **Keypoints Extraction**: MediaPipe extracts keypoints (pose: 33×4, face: 468×3, left hand: 21×3, right hand: 21×3) from each frame, saved as NumPy arrays in the `MP_Data` directory.
- **Data Structure**:
  ```
  MP_Data/
  ├── hello/
  │   ├── 0/
  │   │   ├── 0.npy
  │   │   ├── 1.npy
  │   │   └── ...
  │   └── ...
  ├── thanks/
  └── iloveyou/
  ```

## Model Architecture
The model is an **LSTM-based neural network** designed for sequence classification:
- **Input**: Sequences of keypoints extracted from 30-frame video clips.
- **Layers**:
  - LSTM layers to model temporal dependencies.
  - Dense layers for final classification.
- **Output**: Predicts one of the three actions ("hello," "thanks," "iloveyou").

The model is trained on labeled sequences and evaluated using accuracy metrics, achieving reasonable performance on the test set.

## Results
- **Accuracy**: The model achieves an accuracy of XX% on the test set (update with actual results if available).
- **Confusion Matrix**: Detailed performance can be analyzed using the multilabel confusion matrix generated during evaluation.

## Contribution
Contributions are welcome! To get involved:
- Open an issue to report bugs or suggest features.
- Submit a pull request with your improvements.

---

**Note**: Replace `path/to/your/image.jpg` with the actual path to an image (e.g., a video frame with landmarks drawn using MediaPipe) and update the GitHub URL with your repository link.