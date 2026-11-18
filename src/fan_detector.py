import face_alignment
import cv2
import matplotlib.pyplot as plt

# Initialize FAN
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    device='cpu',
    flip_input=False,
    compile=False
)

# Load image
image_path = "data/images/test.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# Convert BGR to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect facial landmarks
predictions = fa.get_landmarks(rgb_image)

if predictions is None:
    print("No face detected.")
else:
    # Get landmarks for the first detected face
    landmarks = predictions[0]

    # Display and save image
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)

    plt.scatter(
        landmarks[:, 0],
        landmarks[:, 1],
        c='red',
        s=1
    )

    plt.axis("off")

    plt.savefig("results/fan/fan_test_result.jpg", bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.show()