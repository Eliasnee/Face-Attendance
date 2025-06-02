import face_recognition
import os
import pickle

known_encodings = []
known_names = []

for person in os.listdir("known_faces_augmented"):
    person_folder = os.path.join("known_faces", person)
    for img_file in os.listdir(person_folder):
        image_path = os.path.join(person_folder, img_file)
        image = face_recognition.load_image_file(image_path)
        encs = face_recognition.face_encodings(image)
        if encs:
            known_encodings.append(encs[0])
            known_names.append(person)

with open("encodings.pickle", "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print("Encodings generated.")
