# Face Mask Detection (CNN + OpenCV + GUI 🎯)

A real-time **Face Mask Detection System** using **TensorFlow, OpenCV, and Tkinter GUI**.  
It detects whether a person is:

- ✅ With Mask
- ❌ Without Mask
- 😷 Improper Mask  

This project also includes **beep alarm + voice alert**, log system & snapshot saving — a complete deployable solution.

---

## 🚀 Features
| Feature | Description |
|--------|-------------|
📷 Real-time webcam detection | Detect mask status live  
😷 3-class classification | With Mask / Without Mask / Improper Mask  
🔊 Audio Warning | Voice alert: “Please wear a mask”  
🚨 Buzzer | Beep alarm when no mask  
🖼 Capture Images | Saves snapshots folder  
📝 Detection Logs | Auto-logs events with timestamps  
🪟 GUI Interface | Tkinter-based control panel (Start/Stop/Alert buttons)  
📦 Model | Custom CNN trained on Mask dataset  

---

## 🧠 Model Details
- Framework: TensorFlow / Keras
- Architecture: Custom CNN
- Input Size: `224 x 224`
- Loss: `Sparse Categorical Crossentropy`
- Optimizer: `Adam`

---

## 📂 Project Structure


### 🔗 Download Trained Model (.h5)

Download model: https://drive.google.com/file/d/1T9bqXpkARFaoZdM45nHVsYJ-qbDIyVAw/view?usp=sharing
