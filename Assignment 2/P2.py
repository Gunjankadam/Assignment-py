
import cv2


img = cv2.imread('Group-1.jpg')


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



def detect_face(img):
    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.3, minNeighbors=3)

    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return face_img



face_img = detect_face(img)


cv2.imshow('Original Image', img)
cv2.imshow('Detected Face', face_img)
cv2.waitKey(0)
cv2.destroyAllWindows()