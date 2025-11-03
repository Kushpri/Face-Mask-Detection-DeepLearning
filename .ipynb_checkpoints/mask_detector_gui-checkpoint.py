import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import messagebox
from gtts import gTTS
import playsound
import os
import threading
import time

# Load Model
model = load_model("face_mask_detector.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = None
running = False

def speak_alert(text):
    tts = gTTS(text=text, lang='en')
    filename = "alert.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

def mask_detector():
    global cap, running
    cap = cv2.VideoCapture(0)
    
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)
            prediction = model.predict(face, verbose=0)
            
            label = "Mask" if prediction[0][0] > 0.5 else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
            if label == "No Mask":
                threading.Thread(target=speak_alert, args=("Please wear a mask",)).start()

            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 3)
        
        cv2.imshow("Face Mask Detection GUI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def start_camera():
    global running
    running = True
    threading.Thread(target=mask_detector).start()

def stop_camera():
    global running
    running = False

# GUI Window
window = tk.Tk()
window.title("Face Mask Detection GUI")
window.geometry("400x300")

tk.Label(window, text="Face Mask Detector", font=("Arial", 16)).pack(pady=10)
tk.Button(window, text="Start Camera", font=("Arial", 12), command=start_camera).pack(pady=10)
tk.Button(window, text="Stop Camera", font=("Arial", 12), command=stop_camera).pack(pady=10)
tk.Label(window, text="Press 'q' to exit camera window").pack(pady=10)

window.mainloop()
