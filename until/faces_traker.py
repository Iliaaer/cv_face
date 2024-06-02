from scipy.spatial import distance as dist

import until.verifications as vf


class FacesTraker:
    def __init__(self, threshold_new_face: float = 0.2, vf_net: vf.FaceVerification = None,
                 distance_metric: vf.MT = vf.MT.COSINE) -> None:
        self.faces = {}
        self.faces_feature = {}
        self.threshold_new_face = threshold_new_face
        self.faces_name = {}
        self.faces_representation = {}
        self.time = {}
        self.vf_net = vf_net
        self.distance_metric = distance_metric

    def update(self, bboxs: list, face_ids: list, features: list, time: list) -> list:
        for i_element, face_id in enumerate(face_ids):
            is_face = self.faces.get(face_id, False)
            if is_face:
                self.faces[face_id] = bboxs[i_element]
                self.faces_feature[face_id].append(features[i_element])
            else:
                feature_i = features[i_element]
                outputs = {}
                all_faces_id = self.get_faces_id()
                if not all_faces_id:
                    self.faces[face_id] = bboxs[i_element]
                    self.faces_feature[face_id] = [features[i_element]]
                    print(f"[INFO]  New {face_id}")
                    break

                for i_face_element in all_faces_id:
                    faces_features = self.get_faces_feature(face_id=i_face_element)
                    r = dist.cdist(faces_features, [feature_i], 'cosine')
                    outputs[i_face_element] = r.min()
                min_feature = min(outputs.items(), key=lambda unit: unit[1])

                self.faces[face_id] = bboxs[i_element]

                if min_feature[1] <= self.threshold_new_face and min_feature[0] not in face_ids:
                    self.faces_feature[face_id] = self.get_faces_feature(min_feature[0]) + [features[i_element]]
                    self.deregister(face_id=min_feature[0])
                    print(f"[INFO]  New_old {face_id}, DELETE {min_feature[0]}. "
                          f"MIN_feature= {min_feature[1]} face={min_feature[0]}")
                    continue

                self.faces_feature[face_id] = [features[i_element]]
                print(f"[INFO]  New {face_id}. MIN_feature= {min_feature[1]} and face={min_feature[0]}")

        return [self.get_face_bbox(face_id=face_id) + [face_id] for face_id in face_ids]

    def deregister(self, face_id: int) -> None:
        del self.faces[face_id]
        del self.faces_feature[face_id]

    def add(self, face_id: int, other: list) -> None:
        self.faces_representation[face_id] = other

    def get_faces_id(self) -> list:
        return list(self.faces.keys())

    def get_face_bbox(self, face_id: int) -> list:
        return self.faces.get(face_id)

    def get_faces_feature(self, face_id: int) -> list:
        return self.faces_feature.get(face_id)

    def get_minimum_feature(self, face_id: int) -> list:
        if not (face := self.faces_representation.get(face_id)):
            return []
        return min(face, key=lambda x: x[1])

    def add_name(self, face_id: int, name: list) -> None:
        self.faces_name[face_id] = name
        for face_ix in self.get_faces_id():
            representations = self.faces_representation.get(face_ix)
            if face_ix == face_id:
                self.faces_representation[face_ix] = [i_x for i_x in representations if i_x[0] == name[0]]
                continue
            self.faces_representation[face_ix] = [i_x for i_x in representations if i_x[0] != name[0]]

    def __call__(self, threshold: int = 0.38, *args, **kwargs) -> dict:
        for key in self.get_faces_id():
            representation = []
            if not (feature := self.get_faces_feature(key)):
                continue

            for target_representation in feature:
                result_df = self.vf_net.df.copy()
                distances = []
                for index, instance in result_df.iterrows():
                    source_representation = instance[f"{self.vf_net.model_name}_representation"]
                    distance = vf.calculate_distance(
                        source=source_representation,
                        target=target_representation,
                        distance_metric=self.distance_metric
                    )
                    distances.append(distance)

                result_df[f"{self.vf_net.model_name}_{self.distance_metric.name}"] = distances
                threshold = vf.findThreshold(self.vf_net.model_name, self.distance_metric)
                result_df = result_df.drop(columns=[f"{self.vf_net.model_name}_representation"])
                result_df = result_df[result_df[f"{self.vf_net.model_name}_{self.distance_metric.name}"] <= threshold]
                result_df = result_df.sort_values(
                    by=[f"{self.vf_net.model_name}_{self.distance_metric.name}"], ascending=True
                ).reset_index(drop=True)
                if len(result_df):
                    for result_i in range(len(result_df)):
                        candidate = result_df.iloc[result_i]
                        candidate_name = "/".join(candidate["identity"].replace("\\", "/").split("/")[:-1])
                        candidate_distance = candidate[f"{self.vf_net.model_name}_{self.distance_metric.name}"]
                        representation.append(
                            [candidate_name, candidate_distance, candidate["identity"].split("/")[-1]])

            self.add(key, representation)

        min_representation = [None, None, -1]
        while min_representation[2] <= threshold or min_representation[0] is None:
            if min_representation[0]:
                self.add_name(min_representation[0],
                              [min_representation[1], min_representation[2], min_representation[3]])

            representations = [[i] + self.get_minimum_feature(i) for i in self.get_faces_id() if
                               not self.faces_name.get(i, False) and len(self.get_minimum_feature(i)) > 1]
            if not representations:
                break
            min_representation = min(representations, key=lambda x: x[2])

        return {face_id: self.faces_name.get(face_id, None) for face_id in self.faces.keys()}
