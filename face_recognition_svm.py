import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage.feature import hog

# from data_process_svm import get_hog_features


filename = 'model4_(99-99).sav'
loaded_model = pickle.load(open(filename, 'rb'))


def get_hog_features(img, orient=8, pix_per_cell=16, cell_per_block=1,vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualize=vis, feature_vector=feature_vec,multichannel=True)
        return features, hog_image
    else: # Otherwise call with one output
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True, visualize=vis, feature_vector=feature_vec)
                   #,multichannel=True)
        return features


def face_recognition(img, scaleFactor, minNeighbors):

    face_model = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

    faces= face_model.detectMultiScale(img, scaleFactor= scaleFactor, minNeighbors= minNeighbors)

    labels = {0:'Chau', 1:'Khang', 2:'Hieu', 3:'Hoa',4:'Nho',5:'Phuoc'}
    dist_label={0:(0,255,0),1:(255,0,0),2:(0,0,255),3:(255,0,255), 4:(255,255,0),5:(0,255,255)}

    new_size=(99,99)

    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image
        
        
    for i in range(len(faces)):
        (x,y,w,h) = faces[i]

        crop = new_img[y:y+h,x:x+w]
        crop = cv2.resize(crop,new_size)
        # crop = np.reshape(crop,[1,128,128,3])/255

        feat_=[]
        gray_ = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        feat_H=get_hog_features(gray_)
        feat_.append(feat_H)
        pre=np.array(feat_)
            
        result = loaded_model.predict(pre)
        # result = knn.predict(pre)
        print(result)

        cv2.putText(new_img,labels[result[0]],(x, y-10), cv2.FONT_HERSHEY_SIMPLEX,3,dist_label[result[0]],5)
        cv2.rectangle(new_img,(x,y),(x+w,y+h),dist_label[result[0]],5)
            

    plt.figure(figsize=(10,10))
    plt.imshow(new_img)
    plt.axis("off")
    plt.show()

# predict
#img= cv2.imread('nhom22.jpg')
#scaleFactor= 1.25
#minNeighbors= 3

#face_recognition(img, scaleFactor, minNeighbors)
