import numpy as np

def calculate_nme(ground_truth, prediction):
    distances = np.linalg.norm(
        ground_truth - prediction,
        axis=1
    )

    interocular_distance = np.linalg.norm(
        ground_truth[36] - ground_truth[45]
    )

    nme = np.mean(distances) / interocular_distance

    return nme


def calculate_nme_per_region(ground_truth, prediction):
    interocular_distance = np.linalg.norm(
        ground_truth[36] - ground_truth[45]
    )

    distances = np.linalg.norm(
        ground_truth - prediction,
        axis=1
    )

    regions = {
        "face_contour": range(0, 17),
        "eyebrows": range(17, 27),
        "nose": range(27, 36),
        "eyes": range(36, 48),
        "mouth": range(48, 68)
    }

    results = {}

    for region, indices in regions.items():
        region_error = np.mean(distances[list(indices)])
        results[region] = region_error / interocular_distance

    return results
