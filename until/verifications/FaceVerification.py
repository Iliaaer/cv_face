import cv2
import numpy as np
import os
import pickle as pk

import pandas as pd
from tqdm import tqdm

from until.verifications.basemodels import (
    VGGFace,
    Facenet,
    Facenet512,
    DeepFace,
    ArcFace,
    SFace,
    Dlib,
)
from until.verifications.until.transfers import Metric as MT
from until.verifications.until.transfers import VerificationFase as VERIF
from until.verifications.until.functions import changed_face_size, get_target_size, init_folder, get_normalize_image
from until.verifications.until import distance as dst

pd.options.display.float_format = '{:.5f}'.format


def calculate_distance(source, target, distance_metric: MT):
    if distance_metric == MT.COSINE:
        return dst.findCosineDistance(source, target)
    if distance_metric == MT.EUCLIDEAN:
        return dst.findEuclideanDistance(source, target)
    if distance_metric == MT.EUCLIDEAN_L2:
        return dst.findEuclideanDistance(
            dst.l2_normalize(source),
            dst.l2_normalize(target),
        )
    raise ValueError(f"invalid distance metric passes - {distance_metric}")


class FaceVerification:
    def __init__(self, model_name: VERIF = VERIF.FACENET, db_path: str = None, db_reboot: bool = False,
                 db_path_down: str = None):

        models = {
            VERIF.VGGFACE: VGGFace.loadModel,
            VERIF.FACENET: Facenet.loadModel,
            VERIF.FACENET512: Facenet512.loadModel,
            VERIF.DEEPFACE: DeepFace.loadModel,
            VERIF.ARCFACE: ArcFace.loadModel,
            VERIF.SFACE: SFace.load_model,
            VERIF.DLIB: Dlib.DlibClient,
        }

        self.representations = []

        self.model_name = model_name

        print(model_name.name)

        self.model = models.get(model_name)

        if not self.model:
            raise ValueError(f"Invalid model_name passed - {model_name}")

        self.target_size = get_target_size(model_name)

        self.model = self.model()

        self.df = None
        if db_path:
            self.init_db(db_path=db_path, db_reboot=db_reboot)
        if db_path_down:
            self.init_db(db_path_down=db_path_down)

    def __represent(self, image: np.ndarray) -> pd.DataFrame:
        """
        This function represents facial image as vector. The function uses convolutional neural
        networks models to generate vector embeddings.

        Parameters:
                image (np.ndarray): numpy array (BGR)
                encoded images could be passed. Source image can have many faces. Then, result will
                be the size of number of faces appearing in the source image.

        Returns:
                Represent function returns a list of object with multidimensional vector (embedding).
                The number of dimensions is changing based on the reference model.
                E.g. FaceNet returns 128 dimensional vector; VGG-Face returns 2622 dimensional vector.
        """

        face = image.copy()
        face = get_normalize_image(face, self.model_name)
        face = np.expand_dims(face, axis=0)
        return self.model.predict(face)[0]

    def init_db(self, db_path: str = None, db_reboot: bool = False, db_path_down: str = None):
        file_name = f"representations_{self.model_name.name}.pkl".lower()
        if db_path_down:
            if not os.path.isdir(db_path_down):
                raise ValueError("Passed db_path does not exist!")
            if os.path.exists(f"{db_path_down}/{file_name}"):
                with open(f"{db_path_down}/{file_name}", "rb") as f:
                    self.representations = pk.load(f)
                # print(len(self.representations))
            self.df = pd.DataFrame(self.representations, columns=["identity", f"{self.model_name}_representation"])
            return None

        if not os.path.isdir(db_path):
            raise ValueError("Passed db_path does not exist!")

        if db_reboot and os.path.exists(f"{db_path}/{file_name}"):
            os.remove(f"{db_path}/{file_name}")

        employees = []

        for r, _, f in os.walk(db_path):
            for file in f:
                if ".jpg" in file.lower() or ".jpeg" in file.lower() or ".png" in file.lower():
                    exact_path = r + "/" + file
                    employees.append(exact_path)

        if len(employees) == 0:
            raise ValueError(f"There is no image in {db_path} folder! Validate .jpg or .png files exist in this "
                             f"path.")

        if os.path.exists(f"{db_path}/{file_name}"):
            with open(f"{db_path}/{file_name}", "rb") as f:
                self.representations = pk.load(f)

            representations_name = [pf[0] for pf in self.representations]
            for path_file in employees:
                if path_file not in representations_name:
                    # employees_not.append(path_file)
                    self.representations = []
                    os.remove(f"{db_path}/{file_name}")
                    break

        if not os.path.exists(f"{db_path}/{file_name}"):

            pbar = tqdm(
                range(0, len(employees)),
                desc="Finding representations"
            )

            for index in pbar:
                employee = employees[index]

                img_content = changed_face_size(img=cv2.imread(employee),
                                                target_size=self.target_size)
                img_representation = self.__represent(image=img_content)

                self.representations.append([employee, img_representation])

            with open(f"{db_path}/{file_name}", "wb") as f:
                pk.dump(self.representations, f)

        self.df = pd.DataFrame(self.representations, columns=["identity", f"{self.model_name}_representation"])

    def find_image(self, image: np.ndarray, distance_metric: MT = MT.COSINE):
        target_obj = changed_face_size(img=image,
                                       target_size=self.target_size)

        target_representation = self.__represent(image=target_obj)

        result_df = self.df.copy()

        distances = []
        for index, instance in result_df.iterrows():
            source_representation = instance[f"{self.model_name}_representation"]

            distance = calculate_distance(
                source=source_representation,
                target=target_representation,
                distance_metric=distance_metric
            )

            distances.append(distance)

        result_df[f"{self.model_name}_{distance_metric.name}"] = distances
        threshold = dst.findThreshold(self.model_name, distance_metric)
        result_df = result_df.drop(columns=[f"{self.model_name}_representation"])
        # result_df = result_df[result_df[f"{self.model_name}_{distance_metric.name}"] <= threshold]

        result_df = result_df.sort_values(
            by=[f"{self.model_name}_{distance_metric.name}"], ascending=True
        ).reset_index(drop=True)

        return result_df

    def represent_one(self, image: np.ndarray):
        target_obj = changed_face_size(img=image, target_size=self.target_size)
        return self.__represent(image=target_obj)

    def find_representation(self, target_representation, distance_metric: MT = MT.COSINE):
        result_df = self.df.copy()

        distances = []
        for index, instance in result_df.iterrows():
            source_representation = instance[f"{self.model_name}_representation"]

            distance = calculate_distance(
                source=source_representation,
                target=target_representation,
                distance_metric=distance_metric
            )

            distances.append(distance)

        result_df[f"{self.model_name}_{distance_metric.name}"] = distances
        threshold = dst.findThreshold(self.model_name, distance_metric)
        result_df = result_df.drop(columns=[f"{self.model_name}_representation"])
        result_df = result_df[result_df[f"{self.model_name}_{distance_metric.name}"] <= threshold]  # search res dst_min

        result_df = result_df.sort_values(
            by=[f"{self.model_name}_{distance_metric.name}"], ascending=True
        ).reset_index(drop=True)

        return result_df


init_folder()
