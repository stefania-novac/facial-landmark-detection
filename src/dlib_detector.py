import dlib
import cv2


def load_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image


def detect_face(image, detector):
    faces = detector(image, 1)

    if len(faces) == 0:
        raise RuntimeError("No face detected!")

    print(f"Number of faces detected: {len(faces)}")

    return faces[0]


def extract_landmarks(image, face, predictor):
    landmarks = predictor(image, face)

    print(f"Number of landmarks: {landmarks.num_parts}")

    return landmarks


def draw_results(image, face, landmarks):
    cv2.rectangle(
        image,
        (face.left(), face.top()),
        (face.right(), face.bottom()),
        (0, 255, 0),
        2
    )

    for i in range(landmarks.num_parts):
        x = landmarks.part(i).x
        y = landmarks.part(i).y

        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        """
        cv2.putText(
            image,
            str(i),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            1
        )
        """

    return image


def main():
    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor(
        "models/shape_predictor_68_face_landmarks.dat"
    )

    image_path = "data/images/test.jpg"
    output_path = "results/dlib/dlib_test_result.jpg"

    image = load_image(image_path)
    face = detect_face(image, detector)
    landmarks = extract_landmarks(image, face, predictor)

    result = draw_results(image, face, landmarks)

    cv2.imwrite(output_path, result)

    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()
