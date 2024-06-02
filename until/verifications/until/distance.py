import numpy as np
from until.verifications.until.transfers import VerificationFase as VERIF
from until.verifications.until.transfers import Metric as MT


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def findEuclideanDistance(source_representation, test_representation):
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def findThreshold(model_name: VERIF, distance_metric: MT):
    base_threshold = {MT.COSINE: 0.40,
                      MT.EUCLIDEAN: 0.55,
                      MT.EUCLIDEAN_L2: 0.75}

    thresholds = {
        VERIF.VGGFACE: {MT.COSINE: 0.40, MT.EUCLIDEAN: 0.60, MT.EUCLIDEAN_L2: 0.86},
        VERIF.FACENET: {MT.COSINE: 0.40, MT.EUCLIDEAN: 10, MT.EUCLIDEAN_L2: 0.80},
        VERIF.FACENET512: {MT.COSINE: 0.30, MT.EUCLIDEAN: 23.56, MT.EUCLIDEAN_L2: 1.04},
        VERIF.ARCFACE: {MT.COSINE: 0.68, MT.EUCLIDEAN: 4.15, MT.EUCLIDEAN_L2: 1.13},
        VERIF.SFACE: {MT.COSINE: 0.593, MT.EUCLIDEAN: 10.734, MT.EUCLIDEAN_L2: 1.055},
        VERIF.DEEPFACE: {MT.COSINE: 0.23, MT.EUCLIDEAN: 64, MT.EUCLIDEAN_L2: 0.64},
        VERIF.DLIB: {MT.COSINE: 0.07, MT.EUCLIDEAN: 0.6, MT.EUCLIDEAN_L2: 0.4},
    }

    return thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)
