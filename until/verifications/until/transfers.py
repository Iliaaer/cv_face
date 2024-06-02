from enum import Enum


class VerificationFase(Enum):
    VGGFACE: int = 0
    FACENET: int = 1
    FACENET512: int = 2
    DEEPFACE: int = 3
    ARCFACE: int = 4
    SFACE: int = 5
    DLIB: int = 6


class TargetSizeModels(Enum):
    VGGFACE = (224, 224),
    FACENET = (160, 160),
    FACENET512 = (160, 160),
    DEEPFACE = (152, 152),
    ARCFACE = (112, 112),
    SFACE = (112, 112),
    DLIB = (150, 150),


class Metric(Enum):
    COSINE: int = 0
    EUCLIDEAN: int = 1
    EUCLIDEAN_L2: int = 2