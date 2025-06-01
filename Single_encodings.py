import face_recognition
import os
import pickle

def encode_faces_for_person(person_name, input_root="known_faces_augmented", encodings_file="encodings.pickle"):
    input_folder = os.path.join(input_root, person_name)
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"No such folder: {input_folder}")

    known_encodings = []
    known_names = []

    # Load existing encodings if file exists
    if os.path.exists(encodings_file):
        with open(encodings_file, "rb") as f:
            known_encodings, known_names = pickle.load(f)

    new_encodings = []
    new_names = []

    for img_file in os.listdir(input_folder):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(input_folder, img_file)
        image = face_recognition.load_image_file(image_path)
        encs = face_recognition.face_encodings(image)
        if encs:
            new_encodings.append(encs[0])
            new_names.append(person_name)

    # Append new data
    known_encodings.extend(new_encodings)
    known_names.extend(new_names)

    # Save updated encodings
    with open(encodings_file, "wb") as f:
        pickle.dump((known_encodings, known_names), f)

    print(f"✅ Added {len(new_encodings)} encodings for {person_name}. Total now: {len(known_encodings)}")

if __name__ == "__main__":
    encode_faces_for_person("Rita")
