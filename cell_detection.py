import cv2
import numpy as np

cap = cv2.VideoCapture("videos/cokhucre-2.mp4")


def find_center(x, y, w, h):
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return cx, cy


counter = 0
empty_contour = 0
curt_object_track = False
prev_object_track = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi_frame = frame[360:580, 905:1250]
    gray_scale = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray_scale, 5)
    pool = median[25:245, 145:345]
    cv2.line(pool, (50, 0), (50, 220), (0, 0, 255), 2)

    _, threshold = cv2.threshold(pool, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    number_contours = len(contours)
    print("Number of contour:", number_contours)

    if number_contours == 0:
        empty_contour += 1

    for cnt in contours:
        empty_contour = 0
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        print("Area:", area)
        print("x,y,w,h:", x, y, w, h)

        cv2.rectangle(
            roi_frame, (x + 155, y + 25), (x + w + 155, y + h + 25), (0, 255, 0), 2
        )

        cx, _ = find_center(x, y, w, h)

        dist = 50 - cx
        if dist < 0:
            continue
        if dist >= 0:
            curt_object_track = True
        else:
            curt_object_track = False

        if not prev_object_track and curt_object_track:
            counter += 1

        cv2.putText(
            roi_frame,
            str(counter),
            (x + 155, y + 25),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 255, 0),
            2,
        )

    print("Empty contour:", empty_contour)
    print("Prev:", prev_object_track)
    print("Now:", curt_object_track)
    print("Counter:", counter)

    if empty_contour >= 2:
        curt_object_track = False

    prev_object_track = curt_object_track

    cv2.putText(
        frame,
        str(counter),
        (100, 50),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        color=(0, 255, 0),
        thickness=2,
    )

    cv2.imshow("Pool", pool)
    cv2.imshow("Threshold", threshold)
    cv2.imshow("Roi_Frame", roi_frame)

    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
