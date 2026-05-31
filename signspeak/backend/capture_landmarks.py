import cv2
import mediapipe as mp
import csv
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
csv_file = "../dataset/hand_landmarks.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)
cap = cv2.VideoCapture(0)
gesture_name = input("Enter gesture name (e.g., Hello): ")

print("Press 's' to save a frame, 'q' to quit.")

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Capture", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s') and result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            normalized = normalize_landmarks(landmarks)
            data_row = normalized + [gesture_name]
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data_row)
        print("Saved frame!")

cap.release()
cv2.destroyAllWindows()
