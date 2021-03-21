import tensorflow as tf
import cv2
import numpy as np

img_resize = (150,150)
model = tf.keras.models.load_model('./models/mask_detect_model.h5')
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

def mask_predict(face_img):
    """Use model to classify mask """
    face_img = face_img[:,:,::-1] / 255.
    face_img = cv2.resize(face_img,img_resize)
    face_img = np.expand_dims(face_img,0)
    prob = model.predict(face_img)[0][0]
    return prob

def render_predicted_image(img_path):
    """Enhance image with rectangle shape indicating mask"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        prob = mask_predict(face_img)
        if prob < 0.5:
            cv2.putText(img, 'No Mask %.1f%%' % (prob*100), (x, y-5), font, 1, (0, 0, 255), 2)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
        elif prob >= 0.5:
            cv2.putText(img, 'Mask %.1f%%' % (prob*100), (x, y-5), font, 1, (0, 255, 0), 2)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_path = './images/b1.jpg'
    render_predicted_image(img_path)