import pickle

# Load the pickle file
with open("encodings.pickle", "rb") as f:
    known_encodings, known_names = pickle.load(f)

# Now you can use the data
print("Number of encodings loaded:", len(known_encodings))
print("Names:", set(known_names))
