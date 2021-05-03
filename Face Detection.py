import cv2
import mediapipe as mp
import time
pTime = 0
cap = cv2.VideoCapture("G:/Yash/Project/Face Detection/2.mp4")

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawspec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
landmark = 0




while True:
    ret, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawspec, drawspec)

            for lm in faceLms, landmark:
               print(lm)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)















    cv2.imshow("webcam", img)
    k = cv2.waitKey(10)
    if k == 27:
        break
cap.release(1)
cv2.destroyAllWindows()
