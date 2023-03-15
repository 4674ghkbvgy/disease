import cv2
import mediapipe as mp
import joblib
import numpy as np
import time

# Load SVM model
model = joblib.load('svm_model_b.pkl')

# Load MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()
prev_time = 0
curr_time = 0

# Start MediaPipe Hands
np.set_printoptions(precision=6, suppress=True)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

            # Get hand bounding box
            x_min = y_min = 10000
            x_max = y_max = 1
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y

            # Get hand ROI and preprocess for model input
            hand_roi = image[y_min:y_max, x_min:x_max]
            if(hand_roi.shape[0]>0 and hand_roi.shape[1]>0):
                cv2.imshow('roi',hand_roi)
                cv2.waitKey(2)
            if hand_roi.shape[0] < 224 or hand_roi.shape[1] < 224:
                continue
            hand_roi = cv2.resize(hand_roi, (224, 224))
            # hand_roi = np.expand_dims(hand_roi, axis=0)
            
            hand_roi = hand_roi / 255.0
            hand_roi = hand_roi.reshape(1, -1)
            # hand_roi = hand_roi.reshape(hand_roi.shape[0], -1)
            # Predict hand disease
            prediction = model.predict(hand_roi)
            print(prediction)
            if prediction[0] == 1:
                disease = 'Unhealthy'
            else:
                disease = 'Healthy'
                
        # break
            # Draw disease label
            cv2.putText(
                image,
                f'Disease: {disease}',
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA)

    # Calculate and display FPS
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time))
    cv2.putText(image, f'FPS: {fps}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    prev_time = curr_time

    # Show image and handle key press
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Clean up

# Clean up
hands.close()
cap.release()
cv2.destroyAllWindows()
