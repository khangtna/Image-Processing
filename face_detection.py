import cv2
import matplotlib.pyplot as plt


#img= cv2.imread('nhom22.jpg')


def face_detection(img, scaleFactor, minNeighbors):

    face_model = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

    faces= face_model.detectMultiScale(img, scaleFactor = scaleFactor, minNeighbors = minNeighbors)

    out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for i in range(len(faces)):
        (x,y,w,h) = faces[i]

        crop = out_img[y:y+h,x:x+w]
        cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0),5)

    plt.figure(figsize=(10,10))
    plt.imshow(out_img)
    plt.axis("off")
    plt.show()


#face_detection(img)
