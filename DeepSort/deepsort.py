from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2


model = YOLO("yolov8n.pt")  

tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture("cython/assests/vecteezy_people-crossing-the-road-on-zebra-tallin_28257759.mp4")

cv2.namedWindow("Deep SORT Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Deep SORT Tracking", 960, 540)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]


    detections = []
    for det in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = det
        class_name = model.names[int(cls)]
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_name))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    resized = cv2.resize(frame, (960, 540))
    cv2.imshow("Deep SORT Tracking", resized)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
