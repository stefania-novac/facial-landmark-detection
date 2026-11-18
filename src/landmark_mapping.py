import numpy as np

def lerp(a, b, alpha):
    return a + alpha * (b - a)


def wflw_98_to_68(lms_98):
    mapping_98_to_68 = [0, 2, 4, 6, 8, 10, 12, 14, 16,
                        18, 20, 22, 24, 26, 28, 30, 32,
                        33, 34, 35, 36, 37, # left brow
                        42, 43, 44, 45, 46, # right brow
                        51, 52, 53, 54, 55, 56, 57, 58, 59, # nose
                        60, 61, 63, 64, 65, 67, # left eye
                        68, 69, 71, 72, 73, 75, # right eye
                        76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
                        89, 90, 91, 92, 93, 94, 95, # mouth
                        ]
    landmarks_68 = lms_98[mapping_98_to_68, :].copy()

    EYE_INTERP_FACTOR = 0.33
    landmarks_68[37] = lerp(lms_98[61], lms_98[62], EYE_INTERP_FACTOR)
    landmarks_68[38] = lerp(lms_98[63], lms_98[62], EYE_INTERP_FACTOR)
    landmarks_68[40] = lerp(lms_98[65], lms_98[66], EYE_INTERP_FACTOR)
    landmarks_68[41] = lerp(lms_98[67], lms_98[66], EYE_INTERP_FACTOR)

    landmarks_68[43] = lerp(lms_98[69], lms_98[70], EYE_INTERP_FACTOR)
    landmarks_68[44] = lerp(lms_98[71], lms_98[70], EYE_INTERP_FACTOR)
    landmarks_68[46] = lerp(lms_98[73], lms_98[74], EYE_INTERP_FACTOR)
    landmarks_68[47] = lerp(lms_98[75], lms_98[74], EYE_INTERP_FACTOR)

    # landmarks_68[18] = lerp(lms_98[34], lms_98[41], 0.5)
    # landmarks_68[19] = lerp(lms_98[35], lms_98[40], 0.5)
    # landmarks_68[20] = lerp(lms_98[36], lms_98[39], 0.5)
    # landmarks_68[21] = lerp(lms_98[37], lms_98[38], 0.5)

    # landmarks_68[22] = lerp(lms_98[42], lms_98[50], 0.5)
    # landmarks_68[23] = lerp(lms_98[43], lms_98[49], 0.5)
    # landmarks_68[24] = lerp(lms_98[44], lms_98[48], 0.5)
    # landmarks_68[25] = lerp(lms_98[45], lms_98[47], 0.5)     

    return landmarks_68

def mediapipe_468_to_68(landmarks_mp: np.ndarray) -> np.ndarray:
    mp2dlib_correspondence = [
        ## Face Contour
        [127],       # 1
        [234],       # 2
        [93],        # 3
        [132, 58],   # 4
        [58, 172],   # 5
        [136],       # 6
        [150],       # 7
        [176],       # 8
        [152],       # 9
        [400],       # 10
        [379],       # 11
        [365],       # 12
        [397, 288],  # 13
        [288, 361],  # 14
        [323],       # 15
        [454],       # 16
        [356],       # 17

        ## Right Brow 
        [70],        # 18
        [63],        # 19
        [105],       # 20
        [66],        # 21
        [107],       # 22

        ## Left Brow
        [336],       # 23
        [296],       # 24
        [334],       # 25
        [293],       # 26
        [300],       # 27

        ## Nose
        [168, 6],    # 28
        [197, 195],  # 29
        [5],         # 30
        [4],         # 31
        [75],        # 32
        [97],        # 33
        [2],         # 34
        [326],       # 35
        [305],       # 36

        ## Right Eye
        [33],        # 37
        [160],       # 38
        [158],       # 39
        [133],       # 40
        [153],       # 41
        [144],       # 42

        ## Left Eye
        [362],       # 43
        [385],       # 44
        [387],       # 45
        [263],       # 46
        [373],       # 47
        [380],       # 48

        ## Upper Lip Contour Top
        [61],        # 49
        [39],        # 50
        [37],        # 51
        [0],         # 52
        [267],       # 53
        [269],       # 54
        [291],       # 55

        ## Lower Lip Contour Bottom
        [321],   # 56
        [314],   # 57
        [17],   # 58
        [84],   # 59
        [91],   # 60

        ## Upper Lip Contour Bottom
        [78],   # 61
        [82],   # 62
        [13],   # 63
        [312],   # 64
        [308],   # 65

        ## Lower Lip Contour Top
        [317],   # 66
        [14],   # 67
        [87],   # 68
    ]

    for ri in range(68):
        if len(mp2dlib_correspondence[ri]) == 1:
            idx = mp2dlib_correspondence[ri][0]
            mp2dlib_correspondence[ri] = [idx, idx]

    landmarks_mp2dlib = landmarks_mp[mp2dlib_correspondence].mean(axis=1)

    return landmarks_mp2dlib

def mediapipe_468_to_68_v2(landmarks_mp: np.ndarray) -> np.ndarray:

    landmark_points_68 = [
        162, 234, 93, 58, 172, 136, 149, 148, 152,
        377, 378, 365, 397, 288, 323, 454, 389,
        71, 63, 105, 66, 107, 336, 296, 334, 293,
        301, 168, 197, 5, 4, 75, 97, 2, 326, 305,
        33, 160, 158, 133, 153, 144, 362, 385, 387,
        263, 373, 380, 61, 39, 37, 0, 267, 269, 291,
        405, 314, 17, 84, 181, 78, 82, 13, 312, 308,
        317, 14, 87
    ]

    landmarks_68 = landmarks_mp[landmark_points_68].copy()

    return landmarks_68