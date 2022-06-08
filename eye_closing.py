import time
import winsound
import threading
import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import FileVideoStream, VideoStream
from scipy.spatial import distance as dist


def eye_aspect_ratio(eye):
    """Calculates the aspect ratio of the eye
    EAR = (||P2 - P6|| + ||P3 - P5||) / (2 * ||P1 - P4||) 

    Args:
        eye (list): representing the eye position coordinate
    Returns: 
        aspect ratio (float)
    """

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def alert():
    """Produces beeping sound from the speaker to alert the user

    Args:
        None
    Returns:
        None
    """
    for _ in range(3):
        winsound.Beep(1000, 80)
        time.sleep(0.02)
    time.sleep(3)


# Threshold constants
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

CLOSED_TIME_THRESH = 3
ABSENT_TIME_THRESH = 3


closedTime = 0
absentTime = 0
text = 'NOT DETECTED'


# Loading the predictor and the face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# Initializing video stream and the alert thread
vs = FileVideoStream('').start()
fileStream = True
vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)

alertThread = threading.Thread(target=alert)


while True:
    if fileStream and not vs.more():
        break

    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    # Verifying if the face is detected
    if not rects:
        text = 'NOT PRESENT'
        if absentTime == 0:
            absentTime = time.time()
        elif time.time() - absentTime >= ABSENT_TIME_THRESH:
            text = 'ABSENT'
            if not alertThread.is_alive():
                alertThread = threading.Thread(target=alert)
                alertThread.start()
    else:
        absentTime = 0
        text = 'PRESENT'

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extracting the shape of eyes and respective aspect ratios
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        # Marking countours around eyes on the screen
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Checking if the eye is open or closed
        if ear < EYE_AR_THRESH:
            text = 'CLOSED'
            if closedTime == 0:
                closedTime = time.time()
            elif time.time() - closedTime >= CLOSED_TIME_THRESH:
                text = 'SLEEPING'
                if not alertThread.is_alive():
                    alertThread = threading.Thread(target=alert)
                    alertThread.start()
        else:
            closedTime = 0
            text = 'PRESENT'

        cv2.putText( frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()