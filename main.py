import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0)
face_detection_mp = mp.solutions.face_detection

while True:
    ret, frame = cam.read()
    with face_detection_mp.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detector:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = face_detector.process(frame_rgb)
        H, W, _ = frame.shape
        if out.detections is not None:
            for detection in out.detections:
                location_data = detection.location_data
                bbox = location_data.relative_bounding_box
                x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                x1, y1, w, h = int(x1*W), int(y1*H), int(w*W), int(h*H)
                frame[y1:y1+h, x1:x1+w, :] = cv2.blur(frame[y1:y1+h, x1:x1+w, :], (50,50))

    cv2.imshow("Cam Feed", frame)

    if cv2.waitKey(40) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
