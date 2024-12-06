import cv2 #capture image
import pyautogui #input to computer
import mediapipe as mp #recognise hand getsutres
import time

# video capturing
wCam, hCam= 450, 460
prevTime= 0

cap= cv2.VideoCapture(0)

cap.set(3, wCam)
cap.set(4, hCam)

# to recognize hand gestures
mp_hands= mp.solutions.hands
hands= mp_hands.Hands(static_image_mode= False, max_num_hands= 2, min_detection_confidence= 0.7, min_tracking_confidence= 0.5)

# hand land marks on frame
mp_lmark= mp.solutions.drawing_utils


while True:
    ret, frame= cap.read()
    if not ret:
        break

    image_rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result= hands.process(image_rgb)

    # capturing land marks for hand
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_lmark.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # choosing the fingers for volume controling
        index_figer_y= hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        thumb_figer_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

        # pose cases
        if index_figer_y < thumb_figer_y:
            hand_gesture= 'pointing up'
        elif index_figer_y > thumb_figer_y:
            hand_gesture= 'pointing down'
        else:
            hand_gesture= 'other'

        # input cases to computer
        if hand_gesture == 'pointing up':
            pyautogui.press('volumeup')
        elif hand_gesture == 'pointing down':
            pyautogui.press('volumedown')

    currTime= time.time()
    fps= 1/(currTime - prevTime)
    prevTime = currTime

    cv2.putText(frame, f'FPS: {int(fps)}', (40,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0 , 0), 3)
    # displaying camera frame
    cv2.imshow('frame', frame)

    if(cv2.waitKey(1) == ord('q')): #if q pressed the frame is stoped
        break

cap.release()
cv2.destroyAllWindows()