import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Model ve yüz algılama için haarcascade dosyasını yükle
model = load_model('emotion_model.h5')
face_haar_cascade = cv2.CascadeClassifier('xxx.xml')

# Kamera aç
cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()  # Kameradan görüntü al
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # En yüksek olasılıkla tahmin edilen duyguyu bul
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)

    # 'q' tuşuna basılınca döngüyü kır
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows
