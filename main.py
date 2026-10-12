# main.py
# This script is the core of the basketball analytics application.
# It uses a custom YOLOv8 model to detect players and referees,
# and BoT-SORT for tracking to maintain consistent IDs for each person.
# It then calculates and displays real-time statistics on the video feed.

import os
import sys
import math
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Import local utils module with inline fallback
try:
    import utils
except ImportError:
    class _UtilsFallback:
        @staticmethod
        def calculate_distance(point1, point2):
            if point1 is None or point2 is None:
                return 0.0
            return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

        @staticmethod
        def get_player_with_ball(player_positions, ball_position, threshold=150):
            if not player_positions or ball_position is None:
                return None
            closest_player = None
            min_dist = float('inf')
            for track_id, pos in player_positions.items():
                dist = _UtilsFallback.calculate_distance(pos, ball_position)
                if dist < min_dist:
                    min_dist = dist
                    closest_player = track_id
            if min_dist <= threshold:
                return closest_player
            return None

    utils = _UtilsFallback()

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'best.pt') if os.path.exists(os.path.join(SCRIPT_DIR, 'best.pt')) else '/Users/aryantembhurne/Desktop/BTech Project/Project/best.pt'
INPUT_VIDEO_PATH = os.path.join(SCRIPT_DIR, 'game_input.mp4') if os.path.exists(os.path.join(SCRIPT_DIR, 'game_input.mp4')) else '/Users/aryantembhurne/Desktop/BTech Project/Project/game_input.mp4'
OUTPUT_VIDEO_PATH = '/Users/aryantembhurne/Desktop/BTech Project/output_video_new2.mp4'
CONFIDENCE_THRESHOLD = 0.5

# CLASS_NAMES based on your .yaml file
CLASS_NAMES = {
    0: 'Ball', 
    1: 'Hoop', 
    2: 'Period', 
    3: 'Player', 
    4: 'Ref', 
    5: 'Shot Clock', 
    6: 'Team Name', 
    7: 'Team Points', 
    8: 'Time Remaining'
}

# --- Custom Color Mapping for Different Classes ---
# BGR format: (Blue, Green, Red)
COLOR_MAP = {
    'Player': (255, 0, 0),   # Blue for Players
    'Ref': (0, 255, 255),    # Yellow for Referees
    'Ball': (0, 0, 255),     # Red for the Ball
    'default': (0, 255, 0)   # Green for anything else
}

# --- Initialization ---
# Load the custom YOLOv8 model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model from {MODEL_PATH}: {e}")
    sys.exit(1)

# Open the input video file
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file at {INPUT_VIDEO_PATH}")
    sys.exit(1)

# Get video properties for the output file
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0:
    fps = 30  # Default fallback if fps cannot be detected to avoid division by zero

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
if not out.isOpened():
    # Fallback codec if 'mp4v' is unsupported
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# --- Data Storage for Analytics ---
track_history = defaultdict(lambda: [])
player_distance = defaultdict(float)
last_positions = {}
ball_possession_frames = defaultdict(int)

# --- Main Processing Loop ---
frame_number = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video or failed to read frame.")
        break

    frame_number += 1

    # Perform object tracking on the frame
    try:
        results = model.track(frame, persist=True, tracker="botsort.yaml", conf=CONFIDENCE_THRESHOLD, verbose=False)
    except Exception as e:
        print(f"Error during tracking at frame {frame_number}: {e}")
        continue

    # Make a copy of the frame to draw on
    annotated_frame = frame.copy()

    # Safely extract boxes, track IDs, classes, and confidences
    boxes, track_ids, clss, confs = [], [], [], []
    try:
        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xywh.cpu()
            clss = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(boxes)
    except Exception as e:
        print(f"Warning: error parsing detection boxes at frame {frame_number}: {e}")

    # --- Analytics & Manual Drawing ---
    player_positions = {}
    ball_position = None
    player_with_ball = None  # Reset each frame to avoid carrying over stale possession

    # Find the ball's position first
    for i, cls_id in enumerate(clss):
        if CLASS_NAMES.get(cls_id) == 'Ball':
            x, y, _, _ = boxes[i]
            ball_position = (int(x), int(y))
            break

    # Process and draw each detected object
    for box, track_id, cls_id, conf in zip(boxes, track_ids, clss, confs):
        x, y, w, h = box
        center_x, center_y = int(x), int(y)
        class_name = CLASS_NAMES.get(cls_id, 'Unknown')
        
        # Store player positions for possession calculation
        if class_name == 'Player':
            pos_id = track_id if track_id is not None else f"untracked_{len(player_positions)}"
            player_positions[pos_id] = (center_x, center_y)

        # Add current position to history for tracking lines if tracked
        if track_id is not None:
            track_history[track_id].append((center_x, center_y))

        # --- Manual Drawing Logic ---
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        
        # Get color from the color map
        box_color = COLOR_MAP.get(class_name, COLOR_MAP['default'])
        
        # Draw the bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Create and draw the label (ID, class, confidence)
        if track_id is not None:
            label = f"ID:{track_id} {class_name} {conf:.2f}"
        else:
            label = f"{class_name} {conf:.2f}"

        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        # Clamp label y to prevent drawing outside top edge
        label_y = max(y1, label_height + baseline)
        cv2.rectangle(annotated_frame, (x1, label_y - label_height - baseline), (x1 + label_width, label_y), box_color, -1)
        cv2.putText(annotated_frame, label, (x1, label_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # --- Calculate and Display Stats for Players and Refs ---
        if class_name in ['Player', 'Ref'] and track_id is not None:
            if track_id in last_positions:
                dist_moved = utils.calculate_distance(last_positions[track_id], (center_x, center_y))
                player_distance[track_id] += dist_moved
                speed = dist_moved * fps
                
                # Display speed and distance
                stat_y1 = min(y2 + 15, frame_height - 25)
                stat_y2 = min(y2 + 30, frame_height - 10)
                cv2.putText(annotated_frame, f"D: {player_distance[track_id]:.1f}px", 
                            (x1, stat_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"S: {speed:.1f}px/s", 
                            (x1, stat_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            last_positions[track_id] = (center_x, center_y)

        # If you want to draw the tracking lines un-comment the following part of code:
        # if track_id is not None and len(track_history[track_id]) > 1:
        #     points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
        #     cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

    # --- Ball Possession Logic ---
    if ball_position:
        player_with_ball = utils.get_player_with_ball(player_positions, ball_position)
    
    if player_with_ball is not None:
        # Track possession by any player
        ball_possession_frames['Player'] += 1

    # --- Display Statistics on Frame ---
    possession_seconds = ball_possession_frames['Player'] / fps
    cv2.putText(annotated_frame, f"Ball Possession: {possession_seconds:.2f}s", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    out.write(annotated_frame)
    
    try:
        cv2.imshow("Basketball Analytics", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except cv2.error:
        pass

# --- Cleanup ---
cap.release()
out.release()
try:
    cv2.destroyAllWindows()
except cv2.error:
    pass
print(f"Processing complete. Output video saved to {OUTPUT_VIDEO_PATH}")