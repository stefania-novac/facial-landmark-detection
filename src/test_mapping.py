import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import dlib
import mediapipe as mp

from landmark_mapping import wflw_98_to_68, mediapipe_468_to_68, mediapipe_468_to_68_v2

image_path = "data/WFLW/WFLW_images/12--Group/12_Group_Large_Group_12_Group_Large_Group_12_832.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_relative_path = image_path.split("WFLW_images/", 1)[1]

annotation_path = "data/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"


with open(annotation_path, "r") as f:
    lines = f.readlines()

for line in lines:
    values = line.strip().split()

    if values[-1] == image_relative_path:
        break
else:
    raise ValueError("Did not find annotations.")

landmarks_98 = np.array(values[:196], dtype=np.float32).reshape(98, 2)

landmarks_68 = wflw_98_to_68(landmarks_98)

# dlib
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

faces = detector(gray, 1)

if len(faces) == 0:
    raise ValueError("dlib nu a detectat nicio față.")

face = faces[0]

shape = predictor(gray, face)

landmarks_dlib = np.array(
    [[shape.part(i).x, shape.part(i).y] for i in range(68)],
    dtype=np.float32
)

# mediapipe
mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
) as face_mesh:

    results = face_mesh.process(image)

if not results.multi_face_landmarks:
    raise ValueError("MediaPipe nu a detectat nicio față.")

face_landmarks = results.multi_face_landmarks[0]

height, width = image.shape[:2]

landmarks_mp = np.array([
    [point.x * width, point.y * height]
    for point in face_landmarks.landmark
], dtype=np.float32)

landmarks_mp_68 = mediapipe_468_to_68(landmarks_mp)

# plot image
plt.figure(figsize=(10, 10))
plt.imshow(image)

# plt.scatter(
#     landmarks_98[:, 0],
#     landmarks_98[:, 1],
#     s=4,
#     c="yellow",
#     label="WFLW - 98 landmarks"
# )

plt.scatter(
    landmarks_68[:, 0], landmarks_68[:, 1],
    s=4, c="cyan", label="WFLW - 68"
)

plt.scatter(
    landmarks_dlib[:, 0], landmarks_dlib[:, 1],
    s=4, c="red", label="dlib - 68"
)

plt.scatter(
    landmarks_mp_68[:, 0], landmarks_mp_68[:, 1],
    s=4, c="lime", label="MediaPipe - 68"
)

# for i, (x, y) in enumerate(landmarks_68):
#     plt.text(x + 2, y, str(i), color="yellow", fontsize=8)

# for i, (x, y) in enumerate(landmarks_mp_68):
#     plt.text(x + 2, y, str(i), color="lime", fontsize=8)

plt.legend()
plt.axis("off")
plt.show()