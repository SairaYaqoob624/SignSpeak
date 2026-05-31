import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
model = load_model('hand_gesture_model.h5')
csv_file = '../dataset/hand_landmarks.csv'
data = pd.read_csv(csv_file)
labels = data['label'].unique()
le = LabelEncoder()
le.fit(labels)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
cap = cv2.VideoCapture(0)

def normalize_landmarks(landmarks):
    if not landmarks or len(landmarks) != 63:
        return landmarks
    x0, y0, z0 = landmarks[0], landmarks[1], landmarks[2]
    shifted = []
    for i in range(21):
        shifted.append(landmarks[i*3] - x0)
        shifted.append(landmarks[i*3+1] - y0)
        shifted.append(landmarks[i*3+2] - z0)
    max_val = max(abs(val) for val in shifted)
    if max_val > 0:
        return [val / max_val for val in shifted]
    return shifted

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    landmark_list = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])
        if len(landmark_list) == 63:  # 21 landmarks * 3
            normalized = normalize_landmarks(landmark_list)
            X_input = np.array(normalized).reshape(1, -1)
            pred = model.predict(X_input)
            class_id = np.argmax(pred)
            gesture_name = le.inverse_transform([class_id])[0]
            cv2.putText(frame, f'Gesture: {gesture_name}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Hand Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
