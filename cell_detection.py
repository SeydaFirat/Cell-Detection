import cv2
import numpy as np

# Video capture is started.
cap = cv2.VideoCapture("videos/cokhucre-2.mp4")


# Function to find the centre of a cell.
def find_center(x, y, w, h):
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return cx, cy


# Defining trace variables.
counter = 0
empty_contour = 0
curt_object_track = False
prev_object_track = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI (Region of Interest) cuts the relevant part of the video.
    roi_frame = frame[360:580, 905:1250]

    # Grayscale
    gray_scale = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    # Median filtre
    median = cv2.medianBlur(gray_scale, 5)

    # ROI is subtracted from interest
    pool = median[25:245, 145:345]

    # A line is created on the pool screen, the aim is to start counting after this line
    cv2.line(pool, (50, 0), (50, 220), (0, 0, 255), 2)

    # Thresholding is applied to the Pool screen
    _, threshold = cv2.threshold(pool, 200, 255, cv2.THRESH_BINARY)

    # Contour detection part
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    number_contours = len(contours)
    print("Number of contour:", number_contours)
    if number_contours == 0:
        empty_contour += 1

    # Summary
    # First, the condition is created to filter out unwanted small contours and handle only the large and important ones.
    # Secondly, for counting the distance from the centre is calculated, for object tracking the state of the current screen is updated and the conditions under which the object is tracked and the number of contours is increased.
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < 20:
            continue

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

    # Condition that resets the current screen so that the same object is not counted more than once, since the video is split into frames
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

    # Pressing 'q' exits the loop and .waitKey(0) allows to advance the video manually by pressing space bar.
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
