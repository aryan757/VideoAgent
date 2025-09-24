import onnxruntime
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import json
from collections import defaultdict
import time

class PersonReIdentificationTrackerONNX:
    def __init__(self, yolo_model_path="yolov8s.pt", reid_onnx_path="reidentificationnet.onnx", database_folder="person_db"):
        """
        Initialize the Person Re-Identification Tracker with ONNX models.
        
        This class provides a comprehensive system for tracking and re-identifying persons
        in video streams. It combines YOLO for person detection with an ONNX ReID model
        for feature extraction, maintaining both reference persons (known individuals) and
        dynamically tracked persons.
        
        Args:
            yolo_model_path (str): Path to YOLO model file for person detection.
                                 Defaults to "yolov8s.pt"
            reid_onnx_path (str): Path to ONNX ReID model for feature extraction.
                                Defaults to "reidentificationnet.onnx"
            database_folder (str): Directory to store person embeddings and metadata.
                                 Defaults to "person_db"
        
        Logic:
            1. Initializes YOLO model for person detection (class 0)
            2. Sets up ONNX runtime session for ReID feature extraction
            3. Configures image preprocessing pipeline (resize, normalize)
            4. Initializes tracking dictionaries and parameters
            5. Creates database folder if it doesn't exist
            6. Loads existing person database and reference persons
        
        Attributes:
            - similarity_threshold (float): Minimum cosine similarity for person matching (0.8)
            - max_disappeared (int): Frames before removing disappeared tracks (30)
            - max_embeddings_per_person (int): Maximum embeddings stored per person (10)
        """
        self.yolo_model = YOLO(yolo_model_path)
        self.session = onnxruntime.InferenceSession(reid_onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2] if len(input_shape) > 2 else 256
        self.input_width = input_shape[3] if len(input_shape) > 3 else 128
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_height, self.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        self.person_database = {}
        self.person_embeddings_list = {}
        self.active_tracks = {}
        self.next_id = 1
        self.similarity_threshold = 0.8
        self.max_disappeared = 30
        self.disappeared_tracks = defaultdict(int)
        self.max_embeddings_per_person = 10
        self.reference_persons = {}
        self.reference_names = {}
        self.reference_best_similarities = {}
        self.reference_best_snapshots = {}
        self.database_folder = database_folder
        if not os.path.exists(self.database_folder):
            os.makedirs(self.database_folder)
        self.load_database()
    
    def _accumulate_embeddings_from_frame(self, frame, embeddings_accumulator):
        """
        Extract and accumulate ReID embeddings from the first detected person in a frame.
        
        This helper method is used during reference embedding creation to collect
        multiple feature vectors from video frames. It assumes only one person of
        interest is present in the frame.
        
        Args:
            frame (numpy.ndarray): Input video frame (BGR format)
            embeddings_accumulator (list): List to append extracted features to
        
        Returns:
            bool: True if embedding was successfully extracted and added, False otherwise
        
        Logic:
            1. Extract person crops from the frame using YOLO
            2. If no persons detected, return False
            3. Extract ReID features from the first detected person
            4. If feature extraction fails, return False
            5. Append features to accumulator and return True
        
        Note:
            Only processes the first detected person, assuming single-person scenarios
            for reference embedding creation.
        """
        person_crops, _ = self.extract_person_crops(frame)
        if not person_crops:
            return False
        features = self.extract_reid_features(person_crops[0])
        if features is None:
            return False
        embeddings_accumulator.append(features)
        return True
    
    def create_base_embedding_from_folder(self, folder_path, max_frames_per_video=50):
        """
        Create a reference embedding for a person from a folder containing their media.
        
        This method processes all images and videos in a folder to create a comprehensive
        reference embedding for a person. The folder name is used as the person's identifier.
        
        Args:
            folder_path (str): Path to folder containing person's images/videos
            max_frames_per_video (int): Maximum frames to process per video. Defaults to 50
        
        Returns:
            bool: True if reference embedding was successfully created, False otherwise
        
        Logic:
            1. Validate folder existence and extract person name from folder name
            2. Define supported file extensions (images: jpg,jpeg,png,bmp; videos: mp4,avi,mov,mkv,m4v)
            3. Process each file in the folder:
               - For images: Extract features directly from the image
               - For videos: Process frames similar to create_base_embedding_from_video
            4. Accumulate embeddings from all media files
            5. Stop if embedding limit (5 * max_embeddings_per_person) is reached
            6. Compute mean embedding and normalize
            7. Store as reference person and save to disk
        
        Note:
            - Combines embeddings from multiple media types for robust representation
            - Processes videos with frame skipping to avoid redundancy
            - Creates comprehensive person profile from diverse visual samples
        """
        if not os.path.isdir(folder_path):
            print(f"Error: '{folder_path}' is not a directory")
            return False
        person_name = os.path.basename(os.path.normpath(folder_path))
        print(f"Building reference embedding for '{person_name}' from folder: {folder_path}")
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}
        embeddings = []
        entries = sorted(os.listdir(folder_path))
        for entry in entries:
            path = os.path.join(folder_path, entry)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(entry)[1].lower()
            if ext in image_exts:
                img = cv2.imread(path)
                if img is None:
                    continue
                self._accumulate_embeddings_from_frame(img, embeddings)
            elif ext in video_exts:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    print(f"Warning: Could not open video {path}")
                    continue
                frame_count = 0
                processed_frames = 0
                try:
                    while processed_frames < max_frames_per_video:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_count += 1
                        if frame_count % 3 != 0:
                            continue
                        if self._accumulate_embeddings_from_frame(frame, embeddings):
                            processed_frames += 1
                finally:
                    cap.release()
            if len(embeddings) >= 5 * self.max_embeddings_per_person:
                break
        if len(embeddings) == 0:
            print(f"No valid embeddings extracted for {person_name} in folder {folder_path}")
            return False
        embeddings_array = np.array(embeddings)
        mean_embedding = np.mean(embeddings_array, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
        person_id = f"ref_{person_name}"
        self.reference_persons[person_id] = mean_embedding
        self.reference_names[person_id] = person_name
        self.save_reference_person(person_id, person_name, mean_embedding, embeddings)
        print(f"✓ Created reference for '{person_name}' using {len(embeddings)} samples")
        return True
    
    def save_reference_person(self, person_id, person_name, mean_embedding, embeddings_list):
        """
        Save reference person data to disk for persistence across sessions.
        
        This method stores all reference person information including their mean embedding,
        individual embeddings, metadata, and creation timestamp to a pickle file.
        
        Args:
            person_id (str): Unique identifier for the person (format: 'ref_{name}')
            person_name (str): Human-readable name of the person
            mean_embedding (numpy.ndarray): Normalized mean embedding vector
            embeddings_list (list): List of individual embedding vectors
        
        Logic:
            1. Create reference_persons subdirectory if it doesn't exist
            2. Prepare data dictionary with all person information
            3. Add creation timestamp for tracking
            4. Save data to pickle file named '{person_id}.pkl'
            5. Print confirmation of successful save
        
        File Structure:
            database_folder/
            └── reference_persons/
                ├── ref_person1.pkl
                ├── ref_person2.pkl
                └── ...
        
        Note:
            - Uses pickle format for efficient numpy array storage
            - Includes metadata for debugging and tracking
            - Creates organized directory structure
        """
        ref_folder = os.path.join(self.database_folder, "reference_persons")
        if not os.path.exists(ref_folder):
            os.makedirs(ref_folder)
        filepath = os.path.join(ref_folder, f"{person_id}.pkl")
        data = {
            'name': person_name,
            'mean_embedding': mean_embedding,
            'embeddings_list': embeddings_list,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved reference person {person_name} to {filepath}")
    
    def load_reference_persons(self):
        """
        Load previously saved reference persons from disk into memory.
        
        This method reads all reference person pickle files and populates the
        reference_persons and reference_names dictionaries for immediate use.
        
        Logic:
            1. Check if reference_persons directory exists
            2. Find all .pkl files in the directory
            3. For each pickle file:
               - Load the data dictionary
               - Extract person_id from filename
               - Populate reference_persons with mean_embedding
               - Populate reference_names with person name
            4. Handle loading errors gracefully
            5. Print summary of loaded references
        
        Error Handling:
            - Skips corrupted or unreadable files
            - Prints error messages for debugging
            - Continues loading other files if one fails
        
        Note:
            - Called automatically during initialization
            - Enables persistence across application restarts
            - Maintains reference database state
        """
        ref_folder = os.path.join(self.database_folder, "reference_persons")
        if not os.path.exists(ref_folder):
            return
        files = [f for f in os.listdir(ref_folder) if f.endswith('.pkl')]
        for file in files:
            try:
                filepath = os.path.join(ref_folder, file)
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                person_id = file.split('.')[0]
                self.reference_persons[person_id] = data['mean_embedding']
                self.reference_names[person_id] = data['name']
            except Exception as e:
                print(f"Error loading reference person {file}: {e}")
        print(f"Loaded {len(self.reference_persons)} reference persons")
    
    def find_reference_person(self, query_features):
        """
        Find the best matching reference person for given query features.
        
        This method compares query features against all stored reference persons
        using cosine similarity and returns the best match above the threshold.
        
        Args:
            query_features (numpy.ndarray): ReID feature vector to match against references
        
        Returns:
            tuple: (best_match_id, best_similarity, best_name)
                - best_match_id (str): ID of best matching reference person or None
                - best_similarity (float): Cosine similarity score (0.0-1.0)
                - best_name (str): Human-readable name of matched person or None
        
        Logic:
            1. Validate input features and reference database existence
            2. Initialize tracking variables for best match
            3. For each reference person:
               - Compute cosine similarity with query features
               - Update best match if similarity is higher and above threshold
            4. Return best match information or None values if no match
        
        Similarity Calculation:
            - Uses cosine similarity (dot product of normalized vectors)
            - Requires similarity > similarity_threshold (default 0.8)
            - Higher scores indicate better matches
        
        Note:
            - Prioritizes highest similarity match
            - Only returns matches above confidence threshold
            - Returns comprehensive match information
        """
        if not self.reference_persons or query_features is None:
            return None, 0.0, None
        best_match_id = None
        best_similarity = 0.0
        best_name = None
        for person_id, stored_features in self.reference_persons.items():
            similarity = cosine_similarity(
                query_features.reshape(1, -1), 
                stored_features.reshape(1, -1)
            )[0][0]
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match_id = person_id
                best_name = self.reference_names.get(person_id, "Unknown")
        return best_match_id, best_similarity, best_name
    
    def extract_person_crops(self, frame):
        """
        Detect persons in a frame and extract their cropped images with bounding boxes.
        
        This method uses YOLO object detection to find all persons in the input frame
        and returns both the cropped person images and their location information.
        
        Args:
            frame (numpy.ndarray): Input video frame in BGR format
        
        Returns:
            tuple: (person_crops, boxes)
                - person_crops (list): List of cropped person images (numpy arrays)
                - boxes (list): List of bounding box tuples (x1, y1, x2, y2, confidence)
        
        Logic:
            1. Run YOLO prediction on frame with person class filter (class=0)
            2. Set confidence threshold to 0.5 for reliable detections
            3. For each detected person:
               - Extract bounding box coordinates
               - Get detection confidence score
               - Crop person region from original frame
               - Validate crop is not empty
               - Add to results lists
            4. Return synchronized lists of crops and boxes
        
        YOLO Configuration:
            - classes=[0]: Only detect persons (COCO class 0)
            - conf=0.5: Minimum 50% confidence threshold
            - verbose=False: Suppress prediction logs
        
        Note:
            - Crops are extracted as [y1:y2, x1:x2] from original frame
            - Bounding boxes use (x1, y1, x2, y2) format
            - Filters out empty or invalid crops
        """
        results = self.yolo_model.predict(frame, classes=[0], conf=0.5, verbose=False)
        person_crops = []
        boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size > 0:
                    person_crops.append(person_crop)
                    boxes.append((x1, y1, x2, y2, confidence))
        return person_crops, boxes
    
    def extract_reid_features(self, person_crop):
        """
        Extract ReID feature vector from a cropped person image using ONNX model.
        
        This method processes a person crop through the ReID neural network to
        generate a normalized feature vector that represents the person's appearance.
        
        Args:
            person_crop (numpy.ndarray): Cropped person image in BGR format
        
        Returns:
            numpy.ndarray: Normalized feature vector (1D array) or None if extraction fails
        
        Logic:
            1. Validate input crop is not None or empty
            2. Apply preprocessing transformations:
               - Convert BGR to PIL Image
               - Resize to model input dimensions (height x width)
               - Convert to tensor and normalize with ImageNet statistics
            3. Convert tensor to numpy array and ensure float32 type
            4. Run ONNX inference session:
               - Pass preprocessed image through ReID model
               - Extract feature vector from model output
            5. Apply L2 normalization for cosine similarity compatibility
            6. Return flattened normalized features
        
        Preprocessing Pipeline:
            - Resize: To model-specific dimensions (typically 256x128)
            - Normalize: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]
            - Format: Converts to model input format
        
        Error Handling:
            - Returns None for invalid inputs
            - Catches and logs ONNX inference errors
            - Graceful failure without crashing
        
        Note:
            - Features are L2-normalized for consistent similarity computation
            - Uses ImageNet normalization statistics
            - Optimized for person re-identification tasks
        """
        if person_crop is None or person_crop.size == 0:
            return None
        input_tensor = self.transform(person_crop).unsqueeze(0).numpy()
        input_tensor = input_tensor.astype(np.float32)
        try:
            features = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
            norm_features = features / np.linalg.norm(features, axis=1, keepdims=True)
            return norm_features.flatten()
        except Exception as e:
            print(f"Error in ONNX inference: {e}")
            return None
    
    def find_matching_person(self, query_features):
        """
        Find the best matching person from the dynamic tracking database.
        
        This method searches through previously tracked (but not reference) persons
        to find the best match for the query features using cosine similarity.
        
        Args:
            query_features (numpy.ndarray): ReID feature vector to match
        
        Returns:
            tuple: (best_match_id, best_similarity)
                - best_match_id (int): ID of best matching person or None
                - best_similarity (float): Cosine similarity score or 0.0
        
        Logic:
            1. Validate query features and person database existence
            2. Initialize best match tracking variables
            3. For each person in the database:
               - Compute cosine similarity with query features
               - Update best match if similarity is higher and above threshold
            4. Return best match ID and similarity score
        
        Difference from find_reference_person:
            - Searches person_database (dynamic tracks) vs reference_persons (known individuals)
            - Returns person ID (int) vs reference ID (string)
            - Used for re-identifying previously seen but unknown persons
        
        Similarity Requirements:
            - Must exceed similarity_threshold (default 0.8)
            - Uses same cosine similarity calculation as reference matching
            - Prioritizes highest similarity match
        
        Note:
            - Enables re-identification of persons across frames
            - Maintains consistency for non-reference individuals
            - Part of the dynamic tracking system
        """
        if not self.person_database or query_features is None:
            return None, 0.0
        best_match_id = None
        best_similarity = 0.0
        for person_id, stored_features in self.person_database.items():
            similarity = cosine_similarity(
                query_features.reshape(1, -1), 
                stored_features.reshape(1, -1)
            )[0][0]
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match_id = person_id
        return best_match_id, best_similarity
    
    def update_person_database(self, person_id, features):
        """
        Update a person's embedding in the database with new feature observations.
        
        This method implements a rolling average system to continuously improve
        person representations as more observations are collected over time.
        
        Args:
            person_id (int): Unique identifier for the tracked person
            features (numpy.ndarray): New ReID feature vector to incorporate
        
        Logic:
            1. Initialize embeddings list for new persons
            2. Add new features to the person's embedding collection
            3. Maintain rolling window of recent embeddings:
               - Keep only last max_embeddings_per_person (default 10) embeddings
               - Remove oldest embeddings when limit exceeded
            4. Compute new mean embedding from all stored embeddings
            5. Normalize mean embedding for consistent similarity computation
            6. Update person_database with new mean embedding
            7. Save updated embedding to disk for persistence
        
        Rolling Average Benefits:
            - Adapts to appearance changes (lighting, pose, clothing)
            - Reduces impact of poor-quality detections
            - Maintains temporal consistency
            - Improves matching accuracy over time
        
        Storage Management:
            - Limits memory usage by capping embedding count
            - Maintains most recent observations
            - Automatically persists updates to disk
        
        Note:
            - Called whenever a person is successfully re-identified
            - Enables continuous learning and adaptation
            - Balances accuracy with computational efficiency
        """
        if person_id not in self.person_embeddings_list:
            self.person_embeddings_list[person_id] = []
        self.person_embeddings_list[person_id].append(features)
        if len(self.person_embeddings_list[person_id]) > self.max_embeddings_per_person:
            self.person_embeddings_list[person_id] = self.person_embeddings_list[person_id][-self.max_embeddings_per_person:]
        embeddings_array = np.array(self.person_embeddings_list[person_id])
        mean_embedding = np.mean(embeddings_array, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
        self.person_database[person_id] = mean_embedding
        self.save_embedding(person_id)
    
    def save_embedding(self, person_id):
        """
        Save a person's embedding data to disk for persistence.
        
        This method stores both the current mean embedding and the list of
        individual embeddings for a tracked person to a pickle file.
        
        Args:
            person_id (int): Unique identifier for the person to save
        
        Logic:
            1. Construct filepath using person_id (format: 'person_{id}.pkl')
            2. Create data dictionary with:
               - mean_embedding: Current averaged feature vector
               - embeddings_list: List of individual observations
            3. Save dictionary to pickle file using binary write mode
        
        File Organization:
            database_folder/
            ├── person_1.pkl
            ├── person_2.pkl
            └── ...
        
        Data Structure:
            {
                'mean_embedding': numpy.ndarray,
                'embeddings_list': [numpy.ndarray, ...]
            }
        
        Note:
            - Called automatically after database updates
            - Enables persistence across application sessions
            - Uses pickle for efficient numpy array serialization
        """
        filepath = os.path.join(self.database_folder, f"person_{person_id}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump({
                'mean_embedding': self.person_database[person_id],
                'embeddings_list': self.person_embeddings_list.get(person_id, [])
            }, f)
    
    def load_database(self):
        """
        Load previously saved person embeddings and reference persons from disk.
        
        This method restores the complete state of the tracking system including
        both dynamic person database and reference persons from previous sessions.
        
        Logic:
            1. Check if database folder exists, return if not
            2. Find all .pkl files in the database folder
            3. For each pickle file:
               - Extract person_id from filename
               - Load embedding data (mean_embedding and embeddings_list)
               - Handle both dict format (new) and array format (legacy)
               - Track maximum person_id for next_id assignment
            4. Set next_id to max_id + 1 for new person assignment
            5. Load reference persons from subdirectory
            6. Print summary of loaded data
        
        File Format Handling:
            - New format: Dictionary with 'mean_embedding' and 'embeddings_list'
            - Legacy format: Direct numpy array (converted to list format)
        
        ID Management:
            - Maintains continuity of person IDs across sessions
            - Ensures new persons get unique, non-conflicting IDs
        
        Error Handling:
            - Gracefully handles corrupted or unreadable files
            - Continues loading other files if one fails
            - Prints error messages for debugging
        
        Note:
            - Called automatically during initialization
            - Enables seamless session continuity
            - Loads both tracking and reference databases
        """
        if not os.path.exists(self.database_folder):
            return
        files = [f for f in os.listdir(self.database_folder) if f.endswith('.pkl')]
        max_id = 0
        for file in files:
            try:
                person_id = int(file.split('_')[1].split('.')[0])
                filepath = os.path.join(self.database_folder, file)
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        self.person_database[person_id] = data['mean_embedding']
                        self.person_embeddings_list[person_id] = data['embeddings_list']
                    else:
                        self.person_database[person_id] = data
                        self.person_embeddings_list[person_id] = [data]
                if person_id > max_id:
                    max_id = person_id
            except (ValueError, IndexError) as e:
                print(f"Error loading {file}: {e}")
        self.next_id = max_id + 1 if max_id > 0 else 1
        print(f"Loaded {len(self.person_database)} person embeddings from database")
        self.load_reference_persons()
    
    def track_persons_with_reference(self, frame):
        """
        Main tracking method that prioritizes reference persons over dynamic tracking.
        
        This method performs comprehensive person tracking with special handling for
        known reference persons, providing higher priority and confidence for
        identified individuals.
        
        Args:
            frame (numpy.ndarray): Input video frame in BGR format
        
        Returns:
            list: List of detection tuples, each containing:
                (person_id, bbox, similarity, person_name, is_reference)
        
        Logic:
            1. Extract all person crops and bounding boxes from frame
            2. For each detected person:
               a. Extract ReID features from the crop
               b. First, check against reference persons database
               c. If reference match found:
                  - Use reference ID and name
                  - Track best similarity for snapshot saving
                  - Mark as reference detection
               d. If no reference match:
                  - Check against dynamic person database
                  - If match found: re-identify existing person
                  - If no match: create new person ID
                  - Mark as dynamic tracking
            3. Update active tracks with current detections
            4. Handle disappeared tracks (persons no longer visible)
            5. Return list of all current detections
        
        Reference Person Priority:
            - Always checks reference database first
            - Maintains best similarity scores and snapshots
            - Provides stable identification for known individuals
        
        Dynamic Tracking Fallback:
            - Handles unknown persons entering the scene
            - Maintains continuity for non-reference individuals
            - Assigns sequential IDs to new persons
        
        Track Management:
            - Updates last_seen timestamps
            - Maintains active track information
            - Handles track disappearance gracefully
        
        Note:
            - Core method for real-time person tracking
            - Balances reference identification with general tracking
            - Optimized for surveillance and monitoring applications
        """
        person_crops, boxes = self.extract_person_crops(frame)
        current_detections = []
        now = time.time()
        for crop, box in zip(person_crops, boxes):
            features = self.extract_reid_features(crop)
            if features is not None:
                ref_match_id, ref_similarity, ref_name = self.find_reference_person(features)
                if ref_match_id is not None:
                    person_id = ref_match_id
                    person_name = ref_name
                    similarity = ref_similarity
                    is_reference = True
                    prev_best = self.reference_best_similarities.get(ref_match_id, 0.0)
                    if similarity > prev_best:
                        self.reference_best_similarities[ref_match_id] = similarity
                        self.reference_best_snapshots[ref_match_id] = {
                            "frame": frame.copy(),
                            "similarity": similarity,
                            "timestamp": now,
                            "bbox": box
                        }
                else:
                    matched_id, similarity = self.find_matching_person(features)
                    if matched_id is not None:
                        person_id = matched_id
                        person_name = f"Person_{matched_id}"
                        self.update_person_database(person_id, features)
                        if person_id in self.disappeared_tracks:
                            del self.disappeared_tracks[person_id]
                    else:
                        person_id = self.next_id
                        person_name = f"Person_{person_id}"
                        self.next_id += 1
                        self.person_database[person_id] = features
                        self.person_embeddings_list[person_id] = [features]
                        self.save_embedding(person_id)
                    is_reference = False
                self.active_tracks[person_id] = {
                    'bbox': box,
                    'features': features,
                    'last_seen': now,
                    'name': person_name,
                    'is_reference': is_reference
                }
                current_detections.append((person_id, box, similarity, person_name, is_reference))
        self._handle_disappeared_tracks(current_detections)
        return current_detections
    
    def _handle_disappeared_tracks(self, current_detections):
        """
        Manage tracking continuity when persons temporarily disappear from view.
        
        This method implements a grace period system to handle temporary occlusions,
        frame exits, and detection failures without losing track identity.
        
        Args:
            current_detections (list): List of currently detected person IDs
        
        Logic:
            1. Extract current person IDs from detection results
            2. For each active track not in current detections:
               a. Increment disappeared counter for that track
               b. If disappeared count exceeds threshold (max_disappeared):
                  - Remove from active_tracks
                  - Remove from disappeared_tracks
               c. Otherwise, keep track alive for potential re-appearance
        
        Grace Period Benefits:
            - Prevents ID switching during brief occlusions
            - Handles temporary detection failures
            - Maintains track continuity across frame gaps
            - Reduces false track terminations
        
        Threshold Management:
            - max_disappeared: Number of frames before permanent removal (default 30)
            - Balances memory usage with tracking robustness
            - Adjustable based on application requirements
        
        Track Lifecycle:
            1. Active: Person visible and tracked
            2. Disappeared: Person not detected but within grace period
            3. Removed: Grace period exceeded, track permanently deleted
        
        Note:
            - Critical for maintaining stable person identities
            - Handles real-world tracking challenges
            - Prevents database bloat from abandoned tracks
        """
        current_ids = [det[0] for det in current_detections]
        for track_id in list(self.active_tracks.keys()):
            if track_id not in current_ids:
                self.disappeared_tracks[track_id] += 1
                if self.disappeared_tracks[track_id] > self.max_disappeared:
                    del self.active_tracks[track_id]
                    del self.disappeared_tracks[track_id]
    
    def draw_tracking_results(self, frame, detections):
        """
        Visualize tracking results by drawing bounding boxes and labels on the frame.
        
        This method renders the tracking output with color-coded bounding boxes
        and informative labels to provide real-time visual feedback.
        
        Args:
            frame (numpy.ndarray): Input video frame to draw on (modified in-place)
            detections (list): List of detection tuples from tracking methods
        
        Returns:
            numpy.ndarray: Frame with drawn tracking visualizations
        
        Logic:
            1. For each detection in the results:
               a. Extract detection information (ID, bbox, similarity, name, reference status)
               b. Handle both 3-tuple (basic) and 5-tuple (reference-aware) formats
               c. Choose color based on person type:
                  - Red (0,0,255): Reference persons (known individuals)
                  - Green (0,255,0): Dynamic tracked persons (unknown)
               d. Draw bounding box rectangle
               e. Create label with person name and similarity score
               f. Draw label text above bounding box
        
        Visual Elements:
            - Bounding Box: Rectangle around detected person
            - Color Coding: Red for reference, green for tracked
            - Label Format: "{person_name} ({similarity:.2f})"
            - Text Position: Above bounding box (x1, y1-10)
        
        Color Scheme:
            - Reference persons: Red boxes (high importance/known)
            - Tracked persons: Green boxes (standard tracking)
        
        Font Settings:
            - Font: cv2.FONT_HERSHEY_SIMPLEX
            - Scale: 0.6 (readable but compact)
            - Thickness: 2 (clear visibility)
        
        Note:
            - Modifies frame in-place for efficiency
            - Handles both tracking method output formats
            - Provides immediate visual feedback for system performance
        """
        for detection in detections:
            if len(detection) == 3:
                person_id, box, similarity = detection
                person_name = f"Person_{person_id}"
                is_reference = False
            else:
                person_id, box, similarity, person_name, is_reference = detection
            x1, y1, x2, y2, conf = box
            if is_reference:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{person_name} ({similarity:.2f})"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame


def track_person_in_video(person_folder_path, video_path, 
                         yolo_model_path="./models/yolov8s.pt", 
                         reid_onnx_path="./models/resnet50_market1501_aicity156.onnx",
                         database_folder="./database"):
    """
    Simplified function to track a specific person in a video.
    
    This function provides a clean interface where users only need to specify
    the person's folder (containing their reference media) and the video to analyze.
    Everything else is handled automatically.
    
    Args:
        person_folder_path (str): Path to folder containing person's reference images/videos
                                 Example: "./persons/john/" or "/path/to/ashu_folder/"
        video_path (str): Path to the video file to analyze
                         Example: "./video1.mp4" or "/path/to/surveillance_video.mp4"
        yolo_model_path (str): Path to YOLO model. Defaults to "./models/yolov8s.pt"
        reid_onnx_path (str): Path to ReID ONNX model. Defaults to "./models/resnet50_market1501_aicity156.onnx"
        database_folder (str): Directory to store embeddings. Defaults to "./database"
    
    Returns:
        bool: True if tracking completed successfully, False if errors occurred
    
    Logic:
        1. Initialize the Person ReID Tracker with provided model paths
        2. Create reference embedding from the person's folder
        3. Validate the video file can be opened
        4. Extract person name from folder path for celebration messages
        5. Process video frame by frame:
           a. Run person tracking with reference prioritization
           b. Check if the target person is detected
           c. Print celebration message when person is found
           d. Draw tracking results on frame
           e. Display real-time video with overlays
        6. Handle user interruption (press 'q' to quit)
        7. Clean up resources and return success status
    
    Features:
        - Automatic reference embedding creation
        - Real-time person detection celebration
        - Color-coded visualization (red for target person, green for others)
        - Live video display with person identification
        - Graceful error handling and cleanup
    
    Example Usage:
        # Track Ashu in surveillance video
        success = track_person_in_video(
            person_folder_path="./persons/ashu/",
            video_path="./surveillance_video.mp4"
        )
        
        # Track with custom model paths
        success = track_person_in_video(
            person_folder_path="/home/user/people/john/",
            video_path="/home/user/videos/security_cam.mp4",
            yolo_model_path="/models/yolo8n.pt",
            reid_onnx_path="/models/custom_reid.onnx"
        )
    
    Expected Output:
        - Console messages showing reference creation progress
        - Real-time celebration messages when target person is detected
        - Live video window with bounding boxes and person names
        - Final statistics upon completion
    
    Note:
        - Person folder should contain images/videos of the target person
        - Supports common image formats (jpg, png, bmp) and video formats (mp4, avi, mov)
        - Press 'q' key to quit the application
        - Automatically handles model loading and database management
    """
    print(f"🚀 Starting Variphi ReID System")
    print(f"👤 Person folder: {person_folder_path}")
    print(f"🎬 Video file: {video_path}")
    print("-" * 60)
    
    # Initialize tracker
    try:
        tracker = PersonReIdentificationTrackerONNX(
            yolo_model_path=yolo_model_path,
            reid_onnx_path=reid_onnx_path,
            database_folder=database_folder
        )
        print("✅ Tracker initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing tracker: {e}")
        return False
    
    # Create reference embedding from person folder
    print(f"\n📚 Creating reference embedding...")
    success = tracker.create_base_embedding_from_folder(person_folder_path)
    if not success:
        print("❌ Failed to create reference embedding. Check person folder path and contents.")
        return False
    
    # Extract person name from folder path for celebration messages
    person_name = os.path.basename(os.path.normpath(person_folder_path)).lower()
    print(f"✅ Reference created for: {person_name}")
    
    # Open video file
    print(f"\n🎬 Opening video file...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return False
    print("✅ Video opened successfully")
    
    # Get video properties for display
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📊 Video info: {total_frames} frames at {fps} FPS")
    
    print(f"\n🔍 Starting tracking... (Press 'q' to quit)")
    print("=" * 60)
    
    frame_count = 0
    person_detected_frames = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Track persons in current frame
            detections = tracker.track_persons_with_reference(frame)
            
            # Check if target person is detected and print celebration message
            target_detected = False
            for person_id, box, similarity, detected_name, is_reference in detections:
                if is_reference and detected_name.lower() == person_name:
                    if not target_detected:  # Only print once per frame
                        print(f"🎉 YAY, WE FOUND {detected_name.upper()}! Frame {frame_count}, Confidence: {similarity:.3f} 🎉")
                        person_detected_frames += 1
                        target_detected = True
                else:
                    print(f"{person_name.upper()} not detected in frame {frame_count}")   
            
            # Draw tracking results
            result_frame = tracker.draw_tracking_results(frame, detections)
            
            # Add status information to frame
            status_text = f"Frame: {frame_count}/{total_frames} | Target: {person_name.upper()}"
            cv2.putText(result_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if target_detected:
                cv2.putText(result_frame, f"🎉 {person_name.upper()} DETECTED! 🎉", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
            # Display frame
            cv2.imshow(f'Variphi ReID - Tracking {person_name.title()}', result_frame)
            
            frame_count += 1
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⏹️  User requested stop")
                break
                
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during tracking: {e}")
        return False
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "=" * 60)
        print("📊 TRACKING SUMMARY")
        print("-" * 60)
        print(f"👤 Target Person: {person_name.title()}")
        print(f"🎬 Total Frames Processed: {frame_count}")
        print(f"✅ Frames with Target Detected: {person_detected_frames}")
        if frame_count > 0:
            detection_rate = (person_detected_frames / frame_count) * 100
            print(f"📈 Detection Rate: {detection_rate:.1f}%")
        print("=" * 60)
        
    return True


if __name__ == "__main__":
    # Example usage
    print("🔥 Variphi ReID - Simple Person Tracking System 🔥")
    print("\nExample usage:")
    print("track_person_in_video('./persons/ashu/', './reid_testing.mp4'")
    
    # Uncomment and modify these lines to run with your data:
    success = track_person_in_video(
        person_folder_path="./persons/ashu/",
        video_path="./reid_testing.mp4"
    )
    
    if success:
        print("Tracking completed successfully!")
    else:
        print("Tracking failed!")
