import os
from pathlib import Path

import numpy as np
import cv2
from tensorflow.keras.preprocessing import image

from until.verifications.until.transfers import VerificationFase as VERIF
from until.verifications.until.transfers import TargetSizeModels as TSM


def init_folder() -> None:
    """Initialize the folder for storing weights and models.

    Raises:
        OSError: if the folder cannot be created.
        :rtype: object
    """
    home = get_home()
    homePath = get_home_path()
    weightsPath = homePath + "/weights"

    if not os.path.exists(homePath):
        os.makedirs(homePath, exist_ok=True)
        print("Directory ", home, "/.faceVerifications created")

    if not os.path.exists(weightsPath):
        os.makedirs(weightsPath, exist_ok=True)
        print("Directory ", home, "/.faceVerifications/weights created")


def get_home() -> str:
    """Get the home directory for storing weights and models.

    Returns:
        str: the home directory.
    """
    return str(Path.home())


def get_home_path() -> str:
    """Get the home directory for storing weights and models.

    Returns:
        str: the home directory.
    """

    return get_home() + "/.faceVerifications"


def get_target_size(model_name: VERIF) -> tuple:
    """Find the target size of the model.

    Args:
        model_name (str): the model name.

    Returns:
        tuple: the target size.
    """

    target_sizes = {
        VERIF.VGGFACE: TSM.VGGFACE,
        VERIF.FACENET: TSM.FACENET,
        VERIF.FACENET512: TSM.FACENET512,
        VERIF.DEEPFACE: TSM.DEEPFACE,
        VERIF.ARCFACE: TSM.ARCFACE,
        VERIF.SFACE: TSM.SFACE,
        VERIF.DLIB: TSM.DLIB
    }

    target_size = target_sizes.get(model_name).value[0]

    if target_size is None:
        raise ValueError(f"unimplemented model name - {model_name}")

    return target_size


def changed_face_size(
        img,
        target_size=(224, 224)
) -> np.ndarray:
    """Extract faces from an image.

    Args:
        img: numpy array.
        target_size (tuple, optional): the target size of the extracted faces.
        Defaults to (224, 224).

    Returns:
        list: сhanged face size.
    """

    current_img = img.copy()

    if current_img.shape[0] > 0 and current_img.shape[1] > 0:
        factor_0 = target_size[0] / current_img.shape[0]
        factor_1 = target_size[1] / current_img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (
            int(current_img.shape[1] * factor),
            int(current_img.shape[0] * factor),
        )
        current_img = cv2.resize(current_img, dsize)

        diff_0 = target_size[0] - current_img.shape[0]
        diff_1 = target_size[1] - current_img.shape[1]
        current_img = np.pad(
            current_img,
            (
                (diff_0 // 2, diff_0 - diff_0 // 2),
                (diff_1 // 2, diff_1 - diff_1 // 2),
                (0, 0),
            ),
            "constant",
        )

    if current_img.shape[0:2] != target_size:
        current_img = cv2.resize(current_img, target_size)

    img_pixels = image.img_to_array(current_img)
    # img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255  # normalize input in [0, 1]

    return img_pixels.astype(np.float32)


def get_normalize_image(img, model_name: VERIF):
    face = img.copy()
    face *= 255

    if model_name == VERIF.FACENET or model_name == VERIF.FACENET512:
        face = (face - face.mean()) / face.std()

    return face


if __name__ == "__main__":
    img = cv2.imread("../../Face/01/Ilya2.png")
    print(img.shape, type(img))

    cv2.imshow("1", img)

    img2 = cv2.resize(img, (224, 224))
    cv2.imshow("2", img2)

    ef = changed_face_size(img)
    print(ef.shape, type(ef))

    cv2.imshow("3", ef)

    while cv2.waitKey(1) & 0xff != ord('q'):
        pass

    cv2.destroyAllWindows()
