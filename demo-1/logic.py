import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

from collections import defaultdict
import time
from datetime import datetime
import configparser
import keyboard
import numpy as np
import pandas as pd
import faiss
import cv2
from face import *

config = configparser.ConfigParser()
config.read('config.ini')

MODEL_SIZE = config['settings']['model_size']
SKIP_FRAME_ARRAY = list(map(int, config.get('settings', 'skip_frame_array').split(',')))
MIN_SIZE_RATIO = float(config['settings']['min_size_ratio'])
DISTANCE_THRESHOLD = float(config['settings']['distance_threshold'])
print(f"MODEL SIZE: {MODEL_SIZE}")
print(f"SKIP FRAME ARRAY: {SKIP_FRAME_ARRAY}")
print(f"MIN SIZE RATIO: {MIN_SIZE_RATIO}")
print(f"DISTANCE THRESHOLD: {DISTANCE_THRESHOLD}")

DETECTOR = YOLODetection(model='YOLOv11' + MODEL_SIZE.lower() + '-face.pt')
RECOGNIZER = DLIBRecognition()

FAISS_INDEX = None
AVATARS = None
INDEX_TO_ID = []
EMBEDDINGS = []
RECOGNIZED_IDs = set()

# pair with file_name
def read_database():
    avatars = defaultdict(list)
    for file_name in os.listdir('database'):
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            file_path = os.path.join('database', file_name)
            avatar = cv2.imread(file_path)
            avatar = cv2.cvtColor(avatar, cv2.COLOR_BGR2RGB)
            employee_id = file_name.split('-')[0]
            avatars[employee_id].append((avatar, file_name))
    return avatars

print('\nREADING THE DATABASE...')
AVATARS = read_database()

print('\nCREATING VECTOR DATABASE...')
for ID, avatars in AVATARS.items():
    for avatar, file_name in avatars:
        INDEX_TO_ID.append(ID)
        boxes, _ = DETECTOR.detect(avatar)
        if len(boxes) == 0:
            print(f'No face detected in [{file_name}]. Proceeding to embed anyway.')
            left, top, right, bottom = 0, 0, avatar.shape[1], avatar.shape[0]
        else:
            left, top, right, bottom = boxes[0]
        avatar = np.array(avatar[int(top):int(bottom), int(left):int(right)])
        EMBEDDINGS.append(RECOGNIZER.embed(avatar))

EMBEDDINGS = np.array(EMBEDDINGS)
FAISS_INDEX = faiss.IndexFlatL2(EMBEDDINGS.shape[1])
FAISS_INDEX.add(EMBEDDINGS)

index = -1
rec_record = defaultdict(int)
tracker = ByteTracker()
saved_tracker_ids = set()
timekeep_ID = []
timekeep_time = []
cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video path
start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    index += 1
    if SKIP_FRAME_ARRAY[index % len(SKIP_FRAME_ARRAY)] == 0:
        continue
    boxes, scores = DETECTOR.detect(frame)
    face_boxes, conf_scores = [], []
    
    for (left, top, right, bottom), score in zip(boxes, scores):
        if (right - left) * (bottom - top) >= MIN_SIZE_RATIO * frame.shape[0] * frame.shape[1]:
            face_boxes.append((left, top, right, bottom))
            conf_scores.append(score)
    
    tracker_ids, _ = tracker.update(face_boxes, conf_scores)
    queries = []
    
    for i in range(len(tracker_ids)):
        if tracker_ids[i] not in saved_tracker_ids:
            top, right, bottom, left = face_boxes[i]
            queries.append(RECOGNIZER.embed(frame[int(top):int(bottom), int(left):int(right)]))
    if len(queries) == 0:
        continue 
    queries = np.array(queries)
    D, I = FAISS_INDEX.search(queries, 1)
    ids = set()
    tracker.update(boxes, scores)
    
    for i in range(len(queries)):
        if D[i][0] <= DISTANCE_THRESHOLD and ID not in RECOGNIZED_IDs:
            ID = INDEX_TO_ID[I[i][0]]
            ids.add(ID)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            timekeep_ID.append(ID)
            timekeep_time.append(now)
            RECOGNIZED_IDs.add(ID)
            print(f'Recognized employee {ID} at {now}.')

    if keyboard.is_pressed(' '):
        break
end_time = time.time()
print(f'Iterated {index + 1} times in {end_time - start_time} s.')

data = {
    'ID': timekeep_ID,
    'Time': timekeep_time
}
df = pd.DataFrame(data)
print(df)
df.to_csv('data.csv', index=False)