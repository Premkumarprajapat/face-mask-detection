from flask import Flask,render_template,Response
import cv2
import numpy as np
from tensorflow.kera.model import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(_name_)
model= load_model('mask_detector_mode.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascasdes + 'haarcascade_frontalface_default.xml)
labels_dict = {0:'MASK',1:'NO MASK'}
color_dict = {0:(0,255,0),1:(0,0,255)}
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:ret,img = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(imgt,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiscale(gray,1.1,4)

    for x,y,w,h in faces:
        face_img =img[y:y+h,x:x+w]
        resized = cv2.resize(face_img,(224,224))
        normalized = resized/255.0
        reshaped = np.reshape(normalize,(1,224,224,3))
        result = model.predict(reshaped)
        label = np.argumax(result,axis=1)[0]
    
            cv2.rectangle(img, (x, y), (x+w, y+h), color_dict[label], 2)
            cv2.putText(img, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_dict[label], 2)

        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
               
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype = 'multipart/x-mixed-replace;boundary=frame')
if __name__ == '__main__':
    app.run(debug=True
