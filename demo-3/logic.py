import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

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
from insightface.app import FaceAnalysis
from pyzbar.pyzbar import decode

config = configparser.ConfigParser()
config.read('config.ini')

MODEL_SIZE = config['settings']['model_size']
SKIP_FRAME_ARRAY = list(map(int, config.get('settings', 'skip_frame_array').split(',')))
MIN_SIZE_RATIO = float(config['settings']['min_size_ratio'])
SIMILAR_THRESHOLD = float(config['settings']['similar_threshold'])
print(f"MODEL SIZE: {MODEL_SIZE}")
print(f"SKIP FRAME ARRAY: {SKIP_FRAME_ARRAY}")
print(f"MIN SIZE RATIO: {MIN_SIZE_RATIO}")
print(f"SIMILAR THRESHOLD: {SIMILAR_THRESHOLD}")

# Initialize InsightFace buffalo_l model
FACE_APP = FaceAnalysis(name='buffalo_' + MODEL_SIZE.lower())
FACE_APP.prepare(ctx_id=0)

FAISS_INDEX = None
AVATARS = None
INDEX_TO_ID = []
EMBEDDINGS = []
RECOGNIZED_IDs = set()

# Read the database of face embeddings, file_name is employee_id
def read_database():
    avatars = defaultdict(tuple)
    for file_name in os.listdir('database'):
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            employee_id = file_name.split('.')[0]
            file_path = os.path.join('database', file_name)
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            identities = decode(image[:image.shape[0]//3, image.shape[1]*2//3: ,:])
            if(len(identities) == 0):
                print(f'Warning: No QR code detected in [{file_name}]. Skipping this ID card.')
                continue
            full_name = identities[0].data.decode("utf-8").replace('||', '|').split('|')[1]
            faces = FACE_APP.get(image)
            if len(faces) == 0:
                print(f'Warning: No face detected in [{file_name}]. Skipping this ID card.')
                continue
            embedding = faces[0].embedding
            embedding = embedding / np.linalg.norm(embedding)
            avatars[employee_id] = (embedding, full_name)
    return avatars

print('\nREADING THE DATABASE...')
AVATARS = read_database()

print('\nCREATING VECTOR DATABASE...')
EMBEDDINGS = []
INDEX_TO_ID = []

for ID, avatar in AVATARS.items():
    (embedding, full_name) = avatar
    EMBEDDINGS.append(embedding)
    INDEX_TO_ID.append((ID, full_name))

if len(EMBEDDINGS) == 0:
    raise ValueError("No valid embeddings were generated. Check your database images.")

EMBEDDINGS = np.array(EMBEDDINGS)
FAISS_INDEX = faiss.IndexFlatIP(EMBEDDINGS.shape[1])  # Use Inner Product for similarity
FAISS_INDEX.add(EMBEDDINGS)

index = -1
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

    # Extract embeddings directly from the entire frame
    faces = FACE_APP.get(frame)
    
    if len(faces) == 0:
        print("Warning: No face detected in the frame. Skipping this frame.")
        continue
    #else:
        #print(f'Detected {len(faces)} face(s).')
    
    queries = []
    for face in faces:
        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)  # Normalize the embedding
        queries.append(embedding)

    queries = np.array(queries)
    D, I = FAISS_INDEX.search(queries, 1)
    ids = set()

    for i in range(len(queries)):
        ID, full_name = INDEX_TO_ID[I[i][0]]
        if D[i][0] >= SIMILAR_THRESHOLD and ID not in RECOGNIZED_IDs:  # Use >= for Inner Product
            ids.add(ID)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            timekeep_ID.append(ID)
            timekeep_time.append(now)
            RECOGNIZED_IDs.add(ID)
            print(f'Recognized employee {ID}, named {full_name} at {now}.')

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