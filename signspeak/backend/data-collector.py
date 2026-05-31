"""
Data Collection Script for Hand Gestures
Run this to collect training images
"""

import cv2
import os
import time

class DataCollector:
    def __init__(self):
        self.gestures = [
            'hello', 'thank_you', 'yes', 'no', 
            'help', 'please', 'sorry', 'goodbye'
        ]
        self.dataset_path = 'dataset/'
        self.images_per_gesture = 250
        
    def create_folders(self):
        """Create folders for each gesture"""
        for gesture in self.gestures:
            folder_path = os.path.join(self.dataset_path, gesture)
            os.makedirs(folder_path, exist_ok=True)
            print(f"✅ Created folder: {folder_path}")
    
    def collect_data(self):
        """Collect images for all gestures"""
        print("🚀 Starting Data Collection...")
        print("📸 Make sure webcam is connected!")
        print("Press 's' to save image, 'q' to quit gesture, 'ESC' to exit")
        
        self.create_folders()
        cap = cv2.VideoCapture(0)
        
        for gesture in self.gestures:
            print(f"\n🎯 Collecting images for: {gesture.upper()}")
            print(f"📁 Folder: {self.dataset_path}{gesture}/")
            print("Get ready in 5 seconds...")
            time.sleep(5)
            
            count = 0
            gesture_path = os.path.join(self.dataset_path, gesture)
            
            while count < self.images_per_gesture:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Cannot access webcam")
                    break
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Count: {count}/{self.images_per_gesture}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press 's' to save, 'q' to next gesture", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Data Collection - ' + gesture, frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):
                    img_name = f"{gesture}_{count}.jpg"
                    img_path = os.path.join(gesture_path, img_name)
                    cv2.imwrite(img_path, frame)
                    count += 1
                    print(f"✅ Saved: {img_name}")
                
                elif key == ord('q'):
                    print(f"⏩ Skipping {gesture}. Collected {count} images.")
                    break
                
                elif key == 27:  # ESC key
                    print("🛑 Data collection stopped by user")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            
            cv2.destroyWindow('Data Collection - ' + gesture)
        
        cap.release()
        cv2.destroyAllWindows()
        print("🎉 Data collection completed!")

if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_data()