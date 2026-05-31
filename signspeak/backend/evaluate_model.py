
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model # type: ignore
csv_file = '../dataset/hand_landmarks.csv'   # apne dataset path check karo
data = pd.read_csv(csv_file)
X = data.drop('label', axis=1).values
y = data['label'].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)
from tensorflow.keras.utils import to_categorical # type: ignore
y_categorical = to_categorical(y_encoded)
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
model = load_model('hand_gesture_model.h5')
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)
report = classification_report(y_true, y_pred_classes, target_names=le.classes_)
print("\nClassification Report:")
print(report)
