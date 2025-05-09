# -*- coding: utf-8 -*-
"""SignLanguageActionDetection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QZ4Lc9eLOiEnl7C8Iv_WAT9ja2RYDy-Q

# 1. **Import and Install Dependencies**
"""

!pip install mediapipe
!pip install opencv-python-headless

from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode
import cv2
import numpy as np
import mediapipe as mp

# 2. Importing Libraries
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from google.colab.patches import cv2_imshow

"""# New Section"""

# Define the function to process images with Mediapipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Define the function to draw landmarks
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Read the video file
cap = cv2.VideoCapture('WIN_20240422_10_56_53_Pro.mp4')  # replace with your file name

# Your existing code...

# Set up the model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if we've read all frames

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)  # Print the results to debug

        # Check if the model is detecting anything
        if results.pose_landmarks is None and results.face_landmarks is None and results.left_hand_landmarks is None and results.right_hand_landmarks is None:
            print("No detections.")
        else:
            # Draw landmarks
            draw_styled_landmarks(image, results)

            # Show the image
            cv2_imshow(image)

            # Access landmarks here
            if results.left_hand_landmarks:
                print("Number of left hand landmarks:", len(results.left_hand_landmarks.landmark))
            else:
                print("No left hand landmarks detected.")

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

# Your existing code...

# Set up the model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        # Capture image using JavaScript in Colab
        filename = take_photo()

        # Read the image
        frame = cv2.imread(filename)

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)  # Print the results to debug

        # Check if the model is detecting anything
        if results.pose_landmarks is None and results.face_landmarks is None and results.left_hand_landmarks is None and results.right_hand_landmarks is None:
            print("No detections.")
        else:
            # Draw landmarks
            draw_styled_landmarks(image, results)

            # Show the image
            cv2_imshow(image)

            # Access landmarks here
            if results.left_hand_landmarks:
                print("Number of left hand landmarks:", len(results.left_hand_landmarks.landmark))
            else:
                print("No left hand landmarks detected.")

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

# Set up the model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        # Capture image using JavaScript in Colab
        filename = take_photo()

        # Read the image
        frame = cv2.imread(filename)

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)  # Print the results to debug

        # Check if the model is detecting anything
        if results.pose_landmarks is None and results.face_landmarks is None and results.left_hand_landmarks is None and results.right_hand_landmarks is None:
            print("No detections.")
        else:
            # Draw landmarks
            draw_styled_landmarks(image, results)

            # Show the image
            cv2_imshow(image)

            # Access landmarks here
            if results.left_hand_landmarks:
                print("Number of left hand landmarks:", len(results.left_hand_landmarks.landmark))
            else:
                print("No left hand landmarks detected.")

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

from google.colab import output
from IPython.display import display, Javascript, Image
from base64 import b64decode
import cv2
import numpy as np

def take_photo_callback(data_url):
    # Decode the data URL
    binary = b64decode(data_url.split(',')[1])
    with open('photo.jpg', 'wb') as f:
        f.write(binary)
    print('Photo captured and saved as photo.jpg')

# Define the JavaScript function
javascript_code = '''
function takePhoto(quality) {
  return new Promise(async (resolve, reject) => {
    const div = document.createElement('div');
    const capture = document.createElement('button');
    capture.textContent = 'Capture';
    div.appendChild(capture);

    const video = document.createElement('video');
    video.style.display = 'block';
    const stream = await navigator.mediaDevices.getUserMedia({video: true});

    document.body.appendChild(div);
    div.appendChild(video);
    video.srcObject = stream;
    await video.play();

    // Resize the output to fit the video element.
    google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

    // Wait for Capture to be clicked.
    capture.onclick = () => {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      resolve(canvas.toDataURL('image/jpeg', quality));
    };
  });
}

// Run the function to capture a photo
takePhoto(0.8)
  .then(dataUrl => {
    // Send the data URL to Python
    google.colab.kernel.invokeFunction('take_photo_callback', [dataUrl], {});
  })
  .catch(error => console.error(error));
'''

# Execute the JavaScript function
output.register_callback('take_photo_callback', take_photo_callback)
display(Javascript(javascript_code))

len(results.left_hand_landmarks.landmark)

results

draw_landmarks(frame, results)

plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

"""# New Section"""

len(results.left_hand_landmarks.landmark)

pose = []
for res in results.pose_landmarks.landmark:
    test = np.array([res.x, res.y, res.z, res.visibility])
    pose.append(test)

pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    if results.face_landmarks
    else np.zeros(1404)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

result_test = extract_keypoints(results)

result_test

468*3+33*4+21*3+21*3

np.save('0', result_test)

np.load('0.npy')

"""# New Section"""

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# hello
## 0
## 1
## 2
## ...
## 29
# thanks

# I love you

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

"""# New Section"""

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
#                 print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # NEW Apply wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)

                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()

"""# New Section"""

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

label_map = {label:num for num, label in enumerate(actions)}

label_map

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

np.array(sequences).shape

np.array(labels).shape

X = np.array(sequences)

X.shape

y = to_categorical(labels).astype(int)

y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

y_test.shape

"""# New Section"""