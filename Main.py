import cv2 
import pickle
import face_recognition
import numpy as np
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from boxmot.tracker_zoo import create_tracker

# === Setup ===

# Load known encodings
with open("encodings.pickle", "rb") as f:
    known_encodings, known_names = pickle.load(f)

# Load model with explicit task definition
use_onnx = True
model_path = r"yolov8n-face.onnx" if use_onnx else r"yolov8n-face.pt"
model = YOLO(model_path, task='detect')  # Explicitly set task to detect

# Confidence threshold for detection
confidence_threshold = 0.25  # Lower this to detect more, increase to reduce false positives

# Initialize tracker
tracker = create_tracker(
    tracker_type='bytetrack',
    tracker_config='C:\\Users\\OWNER\\boxmot\\boxmot\\configs\\bytetrack.yaml',
    device='cuda'  # or 'cpu'
)

# Initialize camera
cap = cv2.VideoCapture(0)

# Recognition and attendance setup
frame_count = 0
recognition_interval = 10
track_id_to_name = {}         # Maps track_id to name
track_id_recognized_once = {} # Ensures each ID is recognized once
logged_attendance = set()
track_id_candidate_counts = {} # For multi-frame verification
min_verification_frames = 3   # Number of consecutive matches required
recognition_threshold = 0.5   # Face distance threshold (lower is stricter)
min_face_size = 20            # Minimum face size in pixels

# CSV Attendance file
csv_filename = "Attendance.csv"
if not os.path.exists(csv_filename):
    with open(csv_filename, "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Time"])

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame)[0]
    dets = []
    pad = 10

    for box, conf in zip(results.boxes.xyxy, results.boxes.conf):
        if conf < confidence_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
        if (x2 - x1) < min_face_size or (y2 - y1) < min_face_size:
            continue
        dets.append([x1, y1, x2, y2, float(conf), 0])

    tracks = tracker.update(np.array(dets) if dets else np.empty((0, 6)), frame)

    for track in tracks:
        x1, y1, x2, y2, track_id, conf, cls_id, frame_id = track.astype(int)

        # Ensure coordinates are valid and within frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        # Skip if the crop would be empty
        if x1 >= x2 or y1 >= y2:
            continue

        name = track_id_to_name.get(track_id, "Unknown")

        # Run recognition for new or not-yet-identified tracks
        if name == "Unknown":
            run_now = track_id not in track_id_recognized_once or frame_count % recognition_interval == 0

            if run_now:
                face_crop = frame[y1:y2, x1:x2]
                
                # Skip empty or invalid face crops
                if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                    continue
                
                try:
                    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    encs = face_recognition.face_encodings(rgb)
                    
                    if encs:
                        face_distances = face_recognition.face_distance(known_encodings, encs[0])
                        best_match_idx = np.argmin(face_distances)
                        
                        if face_distances[best_match_idx] < recognition_threshold:
                            candidate_name = known_names[best_match_idx]
                            
                            # Multi-frame verification
                            if track_id not in track_id_candidate_counts:
                                track_id_candidate_counts[track_id] = {'name': candidate_name, 'count': 0}
                            
                            if track_id_candidate_counts[track_id]['name'] == candidate_name:
                                track_id_candidate_counts[track_id]['count'] += 1
                            else:
                                # Reset if candidate changed
                                track_id_candidate_counts[track_id] = {'name': candidate_name, 'count': 1}
                            
                            # Verify if we have enough consecutive matches
                            if track_id_candidate_counts[track_id]['count'] >= min_verification_frames:
                                name = candidate_name
                                track_id_to_name[track_id] = name
                                track_id_recognized_once[track_id] = True
                                
                                # Log attendance once per name
                                if name not in logged_attendance:
                                    with open(csv_filename, "a", newline="") as f:
                                        csv.writer(f).writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                                    logged_attendance.add(name)
                        else:
                            # Face doesn't match anyone with sufficient confidence
                            name = "Unknown"
                            if track_id in track_id_candidate_counts:
                                del track_id_candidate_counts[track_id]
                except cv2.error as e:
                    print(f"Skipping frame due to processing error: {e}")
                    continue

        # Draw bounding box and name with confidence indicator
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, red for unknown
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Display verification status if in process
        verification_status = ""
        if name == "Unknown" and track_id in track_id_candidate_counts:
            verification_status = f" (Verifying: {track_id_candidate_counts[track_id]['count']}/{min_verification_frames})"
        
        cv2.putText(frame, name + verification_status, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Face Detection + Recognition + Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()