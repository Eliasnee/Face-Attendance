import os
from Single_cut_frames import extract_faces
from Single_augmentation import augment_faces_for_person
from Single_encodings import encode_faces_for_person

# Map of people to their video paths
person_video_map = {
    "Rita": r"C:\Users\OWNER\Desktop\Videos\WhatsApp Video 2025-05-29 at 19.06.36_073f7ecb.mp4"
}
def register_people(person_video_map):
    for person_name, video_path in person_video_map.items():
        print(f"\n📌 Registering {person_name}...")

        print("➡️  Step 1: Extracting faces")
        extract_faces(name=person_name, video_path=video_path)

        print("➡️  Step 2: Augmenting faces")
        augment_faces_for_person(person_name=person_name)

        print("➡️  Step 3: Encoding faces")
        encode_faces_for_person(person_name=person_name)

        print(f"✅ Completed registration for {person_name}")

if __name__ == "__main__":
    register_people(person_video_map)
