import face_recognition
import os
import pickle
from tqdm import tqdm


known_encodings = []
known_names = []
persons = os.listdir(r"known_faces_augmented")

# Loop through each person with tqdm
for person in tqdm(persons, desc="Processing persons"):
    person_folder = os.path.join("known_faces_augmented", person)


    img_files = os.listdir(person_folder)
    
    # Loop through each image file with tqdm
    for img_file in tqdm(img_files, desc=f"Encoding {person}", leave=False):
        image_path = os.path.join(person_folder, img_file)
        image = face_recognition.load_image_file(image_path)
        encs = face_recognition.face_encodings(image)
        if encs:
            known_encodings.append(encs[0])
            known_names.append(person)

with open("encodings.pickle", "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print("Encodings generated.")
