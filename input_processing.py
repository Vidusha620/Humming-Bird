import cv2
import mediapipe as mp

# Initialize Mediapipe hands and face detection modules
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection

# Initialize DrawingUtils for visualization
mp_drawing = mp.solutions.drawing_utils

# Initialize VideoCapture
cap = cv2.VideoCapture(0)

# Initialize hands and face detection
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands, \
     mp_face.FaceDetection(min_detection_confidence=0.3) as face_detection:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert the frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform hands and face detection
        hands_results = hands.process(rgb_frame)
        face_results = face_detection.process(rgb_frame)

        # Draw hands landmarks on the frame
        if hands_results.multi_hand_landmarks:
            for landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # Draw face bounding box on the frame
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Hand and Face Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()