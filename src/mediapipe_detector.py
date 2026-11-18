import cv2
import mediapipe as mp


def load_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image, rgb_image


def detect_face(rgb_image, face_mesh):
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        raise RuntimeError("No face detected!")

    print(f"Number of faces detected: {len(results.multi_face_landmarks)}")

    return results.multi_face_landmarks[0]


def extract_landmarks(face_landmarks):
    landmarks = face_landmarks.landmark

    print(f"Number of landmarks: {len(landmarks)}")

    return landmarks


def draw_results(image, landmarks):
    height, width = image.shape[:2]

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    return image


def main():
    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    )

    image_path = "data/images/test.jpg"
    output_path = "results/mediapipe/mediapipe_test_result.jpg"

    image, rgb_image = load_image(image_path)
    face_landmarks = detect_face(rgb_image, face_mesh)
    landmarks = extract_landmarks(face_landmarks)

    result = draw_results(image, landmarks)

    cv2.imwrite(output_path, result)

    print(f"Result saved to: {output_path}")

    face_mesh.close()


if __name__ == "__main__":
    main()
