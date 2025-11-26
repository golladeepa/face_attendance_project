import cv2
import os
import pickle
import face_recognition

# Paths
DATASET_DIR = r"C:\Users\Golla\OneDrive\face_attendance_project\datasets\humans"
ENCODINGS_PATH = r"C:\Users\Golla\OneDrive\face_attendance_project\encodings.pickle"

known_encodings = []
known_names = []

print("[INFO] Processing dataset...")

# Loop through folders (each folder = person name)
for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"[INFO] Encoding for {person_name}...")

    # Loop through images for each person
    for image_file in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_file)

        # Load image
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get encodings
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        # Save encodings
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

print("[INFO] Saving encodings to file...")

# Save to pickle
data = {"encodings": known_encodings, "names": known_names}
with open(ENCODINGS_PATH, "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Encoding complete successfully!")
