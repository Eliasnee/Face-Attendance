# Face Recognition Attendance System

## Features
- Real-time face detection using YOLOv8-face
- Face recognition with tracking (ByteTrack)
- Multi-frame verification for accurate identification
- Automatic attendance logging to CSV
- Visual feedback with bounding boxes and verification status


## Installation
1. Clone the boxmot repository
2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/face-attendance-system.git
   cd face-attendance-system


## Install dependencies
   ```bash
   pip install -r requirements.txt
```

### Or manually
   ```bash
   pip install opencv-python numpy face-recognition ultralytics boxmot
```

### Setup Workflow

#### Completed
- [x] **Registration.py**  
  Captures new face samples via webcam/video and organizes them in the dataset

#### Individual Processing
- [ ] **Single_cut_frames.py**  
  Extracts frames from a video for one person
- [ ] **Single_augmentation.py**  
  Generates image variations for a single person
- [ ] **Single_encodings.py**  
  Creates face recognition encodings for one individual

#### Batch Processing 
- [ ] **Augment_all_the_dataset.py**  
  Automatically enhances all registered faces with variations
- [ ] **generate_encodings_for_all_the_dataset.py**  
  Produces recognition encodings for the entire dataset at once


#### Structure of the known faces
known_faces/
   │
   ├── [person_1]/                                   
   │   ├── frame_001.jpg        
   │   ├── frame_002.jpg        
   │   └── ...
   ├── [person_2]/                                   
   │   ├── frame_001.jpg        
   │   ├── frame_002.jpg        
   │   └── ...
   └── ...     


##### Finally run "main.py"

