import cv2
import face_recognition
import pickle
from datetime import datetime
import sqlite3  # <-- added for database

ENCODINGS_PATH = r"C:\Users\Golla\OneDrive\face_attendence_project\encodings.pickle"
ATTENDANCE_FILE = r"C:\Users\Golla\OneDrive\face_attendence_project\attendance.csv"
DB_PATH = r"C:\Users\Golla\OneDrive\face_attendence_project\attendance.db"

# Load encodings
with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# Start webcam
cap = cv2.VideoCapture(0)
print("Webcam started... Press Q to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use HOG model (good for laptops)
    boxes = face_recognition.face_locations(rgb, model="hog")
    encs = face_recognition.face_encodings(rgb, boxes)

    for encoding, box in zip(encs, boxes):

        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.65)
        name = "Unknown"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if True in matches:
            idx = matches.index(True)
