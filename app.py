from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import pickle
from gtts import gTTS
import os
import pygame
import tempfile

app = Flask(__name__)
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)
labels_dict = {0: 'L', 1: 'A', 2: 'B', 3: 'C', 4: 'V', 5: 'W', 6: 'Y', 7: 'He', 8: 'Hello', 9: 'Yes', 10: 'me',
               11: 'Sorry', 12: 'Know', 13: 'Eat', 14: 'You'}

pygame.mixer.init()
previous_prediction = None

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():  # generate frame by frame from camera
    global previous_prediction

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            data_aux = []

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    for landmark in hand_landmarks.landmark:
                        x = landmark.x
                        y = landmark.y
                        data_aux.extend([x, y])

            expected_features = 42
            if len(data_aux) == expected_features:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_value = int(prediction[0])
                if predicted_value in labels_dict:
                    predicted_character = labels_dict[predicted_value]

                    if predicted_character != previous_prediction:
                        previous_prediction = predicted_character

                        text = predicted_character
                        tts = gTTS(text=text, lang='en')

                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                            temp_filename = fp.name

                        tts.save(temp_filename)
                        pygame.mixer.music.load(temp_filename)
                        pygame.mixer.music.play()

                    cv2.putText(frame, f'Sign: {predicted_character}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_audio', methods=['POST'])
def stop_audio():
    pygame.mixer.music.stop()
    return jsonify(success=True)

if __name__ == "__main__":
    app.run(debug=True)