# AI Face Recognition Attendance System

This project is a simple and efficient **Face Recognition-based Attendance System** built using **Python**, **OpenCV**, and **face_recognition** libraries.  
It captures faces from a webcam, matches them with stored face images, and marks attendance in a CSV file — logging the **name** and **datetime** every **30 seconds** if the face is continuously detected.

---

## 📸 Features

- Real-time face detection and recognition.
- Attendance marked with name and timestamp.
- Logs the same face every 30 seconds if still present.
- Automatically creates and updates an `Attendance.csv` file.
- Lightweight and fast — optimized for quick recognition.
- Easy to add new faces — just drop images in the `faces/` folder.

---

## 🛠️ Requirements

- Python 3.7+
- OpenCV
- face_recognition
- numpy

### Install dependencies
```bash
pip install opencv-python face_recognition numpy
