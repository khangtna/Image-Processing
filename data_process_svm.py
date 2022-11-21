import numpy as np
import os
# %matplotlib inline  
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm

from skimage.feature import hog
import cv2
import pickle

# %cd /content/drive/My\ Drive/Share/train_fd

# dataDir='./data/'
dataDir='./dataSet/'
classes=['chau', 'khang', 'hieu', 'hoa','nho','phuoc']
# classes=['chau', 'khang', 'hoa', 'nho']


def statistic(dataDir):
    label = []
    num_images = []
    for lab in os.listdir(dataDir):
        label.append(lab)
        files=os.listdir(os.path.join(dataDir, lab))
        c=len(files)
        num_images.append(c)
    return label, num_images


def LoadData(dataDir,new_size=None):
    if new_size:
        img_rows, img_cols = new_size
    classes=[]
    for _,dirs,_ in os.walk(dataDir):
        classes=dirs
        break  
    num_classes=len(classes)    
    ValidPercent=10
    X_tr=[]
    Y_tr=[]
    X_te=[]
    Y_te=[]    
    for idx,cl in enumerate(classes):
        for _,_,files in os.walk(dataDir+cl+'/'):               
            l=len(files)
            for f in files:
                r=np.random.randint(100)
                img_path=dataDir+cl+'/'+f
                img=cv.imread(img_path)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                if new_size:
                    img=cv.resize(img,(img_rows,img_cols))
                if (r>ValidPercent):
                    X_tr.append(img)
                    Y_tr.append(int(cl[0]))  
                else:
                    X_te.append(img)
                    Y_te.append(int(cl[0]))                      

    return X_tr, Y_tr, X_te, Y_te

# idxs=np.random.permutation(len(img_train))
# plt.figure(figsize = (10,10))
# for i in range(12):  # Lấy ngẫu nhiên 12 mẫu trong tập train
#     idx=idxs[i]
#     plt.subplot(3,4,i+1)
#     plt.imshow(img_train[idx],norm=NoNorm())
#     #plt.title(labels[label_train[idx]])
#     plt.title(label_train[idx])
# plt.show()


# Định nghĩa hàm trích đặc trưng cho từng ảnh
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

# plt.imshow(img_train[0],norm=NoNorm())
# feature, img_hogg =get_hog_features(img_train[0], vis=True)
# plt.imshow(img_hogg,norm=NoNorm())

labels, num_images = statistic(dataDir)
print(labels)
print(num_images)

new_size=(99,99)
img_train,label_train, img_test, label_test=LoadData(dataDir,new_size)


"""
Hiển thị một số thông tin của tập dữ liệu
"""
print("img for train: %d" % (len(img_train)))
print("label for train: %d" % (len(label_train)))

print("img for test: %d" % (len(img_test)))
print("label for test: %d" % (len(label_test)))

#trích đặc trưng cho tập train và test
feat_train=[]
for img in img_train:
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    feat_HOG=get_hog_features(gray)
    feat_train.append(feat_HOG)


feat_test=[]
for img in img_test:
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    feat_HOG=get_hog_features(gray)
    feat_test.append(feat_HOG)

# chuyên qua kiểu numpy
X_hog_tr=np.array(feat_train)
Y_tr=np.array(label_train)
X_hog_te=np.array(feat_test)
Y_te=np.array(label_test)

print("train data: " + str(X_hog_tr.shape))
print("train label: " + str(Y_tr.shape))

print("test data: " + str(X_hog_te.shape))
print("test label: " + str(Y_te.shape))

# Huấn luyện SVM
from sklearn.svm import SVC, LinearSVC
model_svm = SVC(kernel="linear", C=1,gamma='auto', probability=True)
model_svm.fit(X_hog_tr,Y_tr)

# Kiểm thử mô hình SVM
y_predict = model_svm.predict(X_hog_te)
print(y_predict)
print ('Accurary: ',model_svm.score(X_hog_te,Y_te))


# save model
filename = 'model4_(99-99).sav'
pickle.dump(model_svm, open(filename, 'wb'))

