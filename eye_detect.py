import os
from threading import Thread

import cv2
import dlib
import numpy

EAR_THRESHOLD = 0.25
MAX_FRAMES = 30


def play_alarm(path="beep.wav"):
    os.system('vlc --play-and-exit ' + path)


def eye_test(face_mask: numpy.array) -> bool:
    # noinspection SpellCheckingInspection
    """
        Determine if eyes are closed by using Eye Aspect Ratio

        http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

        :type face_mask: numpy.array
        :rtype: bool
        :param face_mask: A 68 point face landmark as returned by dlib.shape_predictor
        :return: Whether eye is closed or not
    """

    # Find Euclidean distance
    def distance(x):
        return numpy.linalg.norm(x)

    def ear(eye):
        return (distance(eye[1] - eye[5]) + distance(eye[2] - eye[4])) / (2 * distance(eye[0] - eye[3]))

    left_eye, right_eye = face_mask[36:42], face_mask[42:48]

    # cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
    # cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

    ear = (ear(left_eye) + ear(right_eye)) / 2
    # cv2.putText(frame, "EAR:" + str(EAR), (10, 13), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    return ear < EAR_THRESHOLD


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("68 face landmarks.dat")

    frame_count = 0

    while True:
        _, frame = cap.read()

        # Convert to gray_scale for faster analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find faces in input
        faces_detected = detector(gray, 0)

        for face in faces_detected:
            # Draw a rectangle around face
            cv2.rectangle(frame, (face.tl_corner().x, (face.tl_corner().y)), (face.br_corner().x, (face.br_corner().y)),
                          (0, 255, 0), 2)

            # Apply 68 landmark model
            faceMask = predictor(gray, face)

            # Face_mask to numpy array
            faceMask = numpy.array([(faceMask.part(x).x, faceMask.part(x).y) for x in range(68)])

            # If test passes consecutively MAX_FRAMES times, trigger alarm
            if eye_test(faceMask):
                frame_count += 1
                if frame_count >= MAX_FRAMES:
                    # New thread to trigger alarm
                    Thread(target=play_alarm, daemon=False).start()
                    frame_count = 0
            else:
                frame_count = 0

        cv2.imshow("WebFeed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
