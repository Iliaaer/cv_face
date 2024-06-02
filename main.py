import asyncio
import threading
from datetime import datetime
import cv2
import numpy as np
from sqlalchemy import select, insert

import until.verifications as vf
from until.sort import Sort
from until.yunet import YuNet
from until.faces_traker import FacesTraker
import until.models as models
import until.schema as schema
from until.database import get_async_session

AUDIENCE = 2614
AUDIENCE_ID = 5
CAMERA_ID = 0

THRESHOLD_FACE_DETECT = 0.80
db_face = "testFace"
distance_metric = vf.MT.COSINE

out: cv2.VideoWriter = None
out_model: cv2.VideoWriter = None
outputFrame: np.ndarray = None
exit_camera: bool = False
exit_main: bool = False
w_size = None
h_size = None
lock = threading.Lock()
loop = asyncio.get_event_loop()

vf_net = vf.FaceVerification(
    db_path=db_face,
)
mot_tracker = Sort(max_age=100, min_hits=10, iou_threshold=0.1)
faces = FacesTraker(
    vf_net=vf_net,
    distance_metric=distance_metric
)

model = YuNet(modelPath="face_detection_yunet_2023mar.onnx",
              inputSize=[320, 320],
              confThreshold=THRESHOLD_FACE_DETECT,
              nmsThreshold=0.3,
              topK=5000,
              backendId=cv2.dnn.DNN_BACKEND_CUDA,
              targetId=cv2.dnn.DNN_TARGET_CUDA)


def camera_main(camera_id: int) -> None:
    global outputFrame, out, exit_camera, model, out_model, w_size, h_size
    cap = cv2.VideoCapture(camera_id)
    w_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_size = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(f'video/main_camera/1.mp4', -1, 20.0, (w_size, h_size))
    out_model = cv2.VideoWriter(f'video/main_camera/2.mp4', -1, 20.0, (w_size, h_size))

    model.setInputSize((w_size, h_size))
    while True:
        ret, image = cap.read()
        if not ret:
            break
        if exit_camera:
            break
        cv2.imshow('main camera', image)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            destroy_main()
            break
        with lock:
            outputFrame = image.copy()

    out.release()
    out_model.release()
    cap.release()
    cv2.destroyWindow('main camera')


def destroy_main() -> None:
    global exit_main
    exit_main = True


def destroy_camera() -> None:
    global exit_camera
    exit_camera = True


def get_image() -> np.ndarray:
    with lock:
        return outputFrame


def wait_camera() -> bool:
    while True:
        image = get_image()
        try:
            if image.shape:
                return True
            break
        except:
            pass
    return True


is_redy_student_id: bool = False
result_get_student_id: int = None


async def get_student_id(path: str):
    global is_redy_student_id, result_get_student_id
    path = "/".join(path.split('/')[1:])
    query = select(models.PathStudetnsFiles.student_id).where(models.PathStudetnsFiles.path == path)
    async with get_async_session() as session:
        result_id = await session.execute(query)

    result_all = result_id.all()
    is_redy_student_id = True
    if len(result_all) == 0:
        result_get_student_id = None
        return []
    result_get_student_id = result_all[0][0]


is_redy_discipline_id: bool = False
result_get_discipline_id: int = None


async def get_discipline_id(date, time):
    global is_redy_discipline_id, result_get_discipline_id
    global AUDIENCE_ID

    schedule = {
        1: ("9:00", "10:30"),
        2: ("10:40", "12:10"),
        3: ("12:20", "13:50"),
        4: ("14:30", "16:00"),
        5: ("16:10", "17:40")
    }

    time_ok = None
    for i, (start, end) in schedule.items():
        if start <= time <= end:
            time_ok = schedule.get(i)[0]

    if not time_ok:
        time_ok = min(schedule.values(), key=lambda x: (x[0] >= time, x[0]))

    query = select(models.Discipline.id).where(models.Discipline.audience_id == AUDIENCE_ID).where(
        models.Discipline.data == date).where(models.Discipline.time == time_ok)
    async with get_async_session() as session:
        result_id = await session.execute(query)

    result_all = result_id.all()
    is_redy_discipline_id = True
    if len(result_all) == 0:
        result_get_discipline_id = None
        return []
    result_get_discipline_id = result_all[0][0]


async def post_attendance(student_id, discipline_id, time):
    attendance = schema.Attendance
    attendance.time = time
    attendance.student_id = student_id
    attendance.discipline_id = discipline_id
    query = insert(models.Attendance).values(**attendance.dict())
    async with get_async_session() as session:
        await session.execute(query)
        await session.commit()
    print(attendance)


if __name__ == "__main__":
    main_camera = threading.Thread(target=camera_main, args=("video/2.mp4",))

    main_camera.start()
    wait_camera()
    while True:
        if exit_main:
            break
        frame = get_image().copy()

        results = model.infer(frame)

        faces_detections = []
        for det in results:
            bbox = det[0:4].astype(np.int32).tolist()
            conf = det[-1]
            faces_detections.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], conf])

        if faces_detections:
            faces_detections = np.array(faces_detections)
        else:
            faces_detections = np.empty((0, 5))

        track_bbs_ids = mot_tracker.update(np.array(faces_detections))

        faces_id = []
        faces_bbox = []
        faces_feature = []
        for d in track_bbs_ids:
            d = d.astype(np.int32)
            x1, y1, x2, y2, object_id = d

            x1, x2 = max(0, min(w_size, x1)), max(0, min(w_size, x2))
            y1, y2 = max(0, min(h_size, y1)), max(0, min(h_size, y2))
            object_id = int(object_id)

            face_one = frame[y1:y2, x1:x2]

            faces_bbox.append([x1, y1, x2, y2])
            faces_feature.append(vf_net.represent_one(face_one).astype(np.float64).tolist())
            faces_id.append(object_id)

        output = faces.update(bboxs=faces_bbox, face_ids=faces_id, features=faces_feature)

        for d in output:
            x1, y1, x2, y2, object_id = d
            text = f"ID {object_id}"
            c_x = int((x1 + x2) / 2)
            c_y = int((y1 + y2) / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(frame, (c_x, c_y), 4, (0, 255, 0), -1)
            cv2.putText(frame, text, (c_x - 10, c_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out_model.write(frame)

        cv2.imshow('result', frame)

        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            destroy_camera()
            break

    main_camera.join()
    cv2.destroyWindow('result')

    result = faces()

    print(result)

    for key, value in result.items():
        if value:
            print(f"[KEY] {key} = {value}")

            is_redy_student_id = False
            loop.run_until_complete(get_student_id(value[0]))
            while not is_redy_student_id:
                pass
            if not result_get_student_id:
                continue
            date_now = datetime.now().date()
            time_now = datetime.now().time().strftime("%H:%M:00")
            print(value[0], result_get_student_id, date_now, time_now)
            is_redy_discipline_id = False
            loop.run_until_complete(get_discipline_id(date_now, time_now))
            while not is_redy_discipline_id:
                pass
            if not result_get_discipline_id:
                continue
            loop.run_until_complete(post_attendance(result_get_student_id, result_get_discipline_id, time_now))
