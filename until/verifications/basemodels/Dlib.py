import os
import bz2
import gdown
import numpy as np
import dlib

from until.verifications.until import functions


class DlibResNet:
    def __init__(self):
        self.layers = [DlibMetaData()]

        home = functions.get_home_path()
        weight_file = home + "/weights/dlib_face_recognition_resnet_model_v1.dat"

        if not os.path.isfile(weight_file):
            file_name = "dlib_face_recognition_resnet_model_v1.dat.bz2"
            url = f"http://dlib.net/files/{file_name}"
            output = f"{home}/weights/{file_name}"
            gdown.download(url, output, quiet=False)

            zipfile = bz2.BZ2File(output)
            data = zipfile.read()
            new_file_path = output[:-4]
            with open(new_file_path, "wb") as f:
                f.write(data)

        self.model = dlib.face_recognition_model_v1(weight_file)


class DlibClient:
    def __init__(self):
        self.model = DlibResNet().model

    def predict(self, img_aligned):
        if len(img_aligned.shape) == 4:
            img_aligned = img_aligned[0]

        img_aligned = img_aligned[:, :, ::-1]  # bgr to rgb
        if img_aligned.max() <= 1:
            img_aligned = img_aligned * 255

        img_aligned = img_aligned.astype(np.uint8)

        img_representation = self.model.compute_face_descriptor(img_aligned)
        img_representation = np.array(img_representation)
        img_representation = np.expand_dims(img_representation, axis=0)
        return img_representation


class DlibMetaData:
    def __init__(self):
        self.input_shape = [[1, 150, 150, 3]]
