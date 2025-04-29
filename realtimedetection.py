from tensorflow.keras.models import load_model
import cv2, numpy as np
# 1) load model (architecture + weights in one file)
model = load_model('facialemotionmodel.h5')

# 2) labels
emotion_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

# 3) face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 4) webcam loop (same as before)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48)).astype('float32')/255.0
        roi = np.expand_dims(roi, axis=(0,-1))
        preds = model.predict(roi)
        label = emotion_labels[np.argmax(preds)]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, .9, (255,0,0), 2)
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1)&0xFF==ord('q'): break
cap.release()
cv2.destroyAllWindows()
