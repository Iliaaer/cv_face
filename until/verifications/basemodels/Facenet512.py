import os
import gdown
from until.verifications.basemodels import Facenet
from until.verifications.until import functions


def loadModel(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5",
):

    model = Facenet.InceptionResNetV2(dimension=512)

    home = functions.get_home_path()
    output = home + "/weights/facenet512_weights.h5"

    if not os.path.isfile(output):
        print("facenet512_weights.h5 will be downloaded...")

        gdown.download(url, output, quiet=False)

    model.load_weights(output)

    return model
