import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import data_process_svm
import face_detection
import face_recognition_svm
from face_recognition_svm import face_recognition



DIR_PATH = r'C:\Users\Hieu Le\Desktop\ImageProcess\anhsauxuly'

# Thay đổi đường dẫn file.


# Menu thao tác
def menu():
    print('-----------------------------------')
    print('Ban muon thuc hien thao tac gi ?'
          '\n- Loc nhieu: Nhap so 1'
          '\n- Tang sac net: Nhap so 2'
          '\n- Lam min anh: Nhap so 3'
          '\n- Tang, giam tuong phan: Nhap so 4'
          '\n- Tang, giam do sang: Nhap so 5'
          '\n- Tu dong tang cuong anh: Nhap so 6'
          '\n- Xac dinh edge cua anh: Nhap so 7'
          '\n- Tim kiem khuon mat: Nhap so 8'
          '\n- Nhan dang khuon mat: Nhap so 9'
          '\n- Thoat: Nhap so 10')
    print('-----------------------------------')
    return


# Kiểm tra  lựa chọn trong menu có nằm đúng trong phạm vi 1 - 8 không
def checkLuaChon(n):
    check = False
    if n not in range(1, 11):
        check = True
    return check


# Lựa chọn trên menu chỉ được nhận số nguyên dương
def readOnlyNumberMenu():
    while True:
        try:
            n = int(input('Phuong an ban chon la: '))
            while checkLuaChon(n):
                n = int(input('Phuong an ban chon khong ton tai, hay nhap lai: '))
            break
        except:
            print('Gia tri nhap vao khong hop le')
    return n


# Kiểm tra lựa chọn phương án lọc ảnh có nằm đúng phạm vi 1 - 3 không
def checkPhuongAn(m):
    checkPa = False
    if m not in range(1, 4):
        checkPa = True
    return checkPa


# Hàm nhập phương án lọc nhiễu chỉ chấp nhận số
def readOnlyNumber():
    while True:
        try:
            pa = int(input('Phuong an ban chon la: '))
            while checkPhuongAn(pa):
                pa = int(input('Lua chon cua ban khong ton tai, hay nhap lai: '))
            break
        except:
            print('Gia tri nhap vao khong hop le')
    return pa


# In ảnh ra màn hình
def showAnh(img, result):
    fig = plt.figure(figsize=(16, 9))
    ax1, ax2 = fig.subplots(1, 2)

    rgb1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ax1.imshow(rgb1)
    ax1.set_title("Anh goc")
    ax1.axis("off")

    rgb2 = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    ax2.imshow(rgb2)
    ax2.set_title("Anh sau khi xu li")
    ax2.axis("off")

    plt.show()
    return


# Tăng giảm tương phản, độ sáng
def contrast_brightness(img, tp, ds):
    contrast_img = cv2.addWeighted(img, 1 + tp, np.zeros(img.shape, img.dtype), 0, 0 + ds)
    return contrast_img


# Tăng sắc nét
def sharpness2(img, xsn):
    gaussian_img = cv2.GaussianBlur(img, (7, 7), 0)
    sharpness_img = cv2.addWeighted(img, 5.5 + xsn, gaussian_img, -4.5 - xsn, 0)
    # Crp 5.5 -4.5
    # 3.5 -2.5
    # 6.5 -5.5
    return sharpness_img


# Bộ lọc tích chập hai chiều (2D Convolution) - làm mượt ảnh
def smoothing(img, xlm):
    kernel = np.ones((xlm, xlm), np.float32) / (xlm * xlm)
    imgSmooth = cv2.filter2D(img, -1, kernel)
    return imgSmooth


# Bộ lọc Gaussian - được sử dụng nhiều
def gaussian(img, x, y, z):
    gaussian_img = cv2.GaussianBlur(img, (x, y), z)
    return gaussian_img


# Bộ lọc median - lọc trung vị - lọc nhiễu nhưng phá cả biên
def median(img, ksize):
    median_img = cv2.medianBlur(img, ksize)
    return median_img


# Bộ lọc bilateral - lọc nhiễu + bảo vệ viền
def bilateral(img, d, m, n):
    blur = cv2.bilateralFilter(img, d, m, n)  # 7
    # Lớn hơn 150 mới có tác dụng, phần mềm offline k cần tg thực thì set 9
    return blur


# Tìm egde
def timEdge(img, nthap, ncao):
    edge = cv2.Canny(img, nthap, ncao)
    return edge


# Phương án lọc nhiễu Gaussian
def locNhieu1(img):
    print('-----------------------------')
    print('Bo loc nhieu Gaussian')
    print('Gia tri ban nhap vao phai la:'
          '\n- stigmaX, stigmaY, ksize la so le nguyen duong.'
          '\n- stigmaX, stigmaY thuoc doan [3, 11].'
          '\n- ksize thuoc doan [0, 3].')
    print('-----------------------------')

    # Chỉ cho nhập một số x
    while True:
        try:
            x = int(input('Nhap vao gia tri stigmaX: '))
            while x % 2 == 0 or x not in range(3, 12):
                x = int(input('Gia tri x sai dieu kien, nhap lai: '))
            break
        except:
            print('Gia tri nhap vao khong hop le')

    # Chỉ cho nhập một số y
    while True:
        try:
            y = int(input('Nhap vao gia tri stigmaY: '))
            while y % 2 == 0 or y not in range(3, 12):
                y = int(input('Gia tri y sai dieu kien, nhap lai: '))
            break
        except:
            print('Gia tri nhap vao khong hop le')

    # Chỉ cho nhập một số z
    while True:
        try:
            z = int(input('Nhap vao gia tri ksize: '))
            if z == 0: break
            else:
                while z % 2 == 0 or z not in range(0, 4):
                    z = int(input('Gia tri z sai dieu kien, nhap lai: '))
                    if z == 0:
                        break
            break
        except:
            print('Gia tri nhap vao khong hop le')

    result_gau = gaussian(img, x, y, z)

    return result_gau


# Phương án lọc nhiễu Median
def locNhieu2(img):
    print('-----------------------------')
    print('Bo loc nhieu Median')
    print('Gia tri ban nhap vao phai la:'
          '\n- La mot so le nguyen duong.'
          '\n- Thuoc doan [5, 11].')
    print('-----------------------------')

    # Chỉ cho nhập một số
    while True:
        try:
            ksize = int(input('Nhap kich co cua ma tran loc trung vi: '))
            while ksize % 2 == 0 or ksize not in range(5, 12):
                ksize = int(input('Gia tri sai dieu kien, hay nhap lai: '))
            break
        except:
            print('Gia tri nhap vao khong hop le')

    result_med = median(img, ksize)

    return result_med


# Phương án lọc nhiễu Bilateral
def locNhieu3(img):
    print('-----------------------------')
    print('Bo loc nhieu Bilateral')
    print('Gia tri ban nhap vao phai la:'
          '\n- D La mot so le nguyen duong.'
          '\n- D thuoc doan [3, 9].'
          '\n- M, N thuoc doan [1, 255].')
    print('-----------------------------')

    # Chỉ cho nhập một số d
    while True:
        try:
            d = int(input('Nhap thong so d cho bo loc: '))
            while d % 2 == 0 or d not in range(3, 10):
                d = int(input('Gia tri d sai dieu kien, hay nhap lai: '))
            break
        except:
            print('Gia tri nhap vao khong hop le')

    # Chỉ cho nhập một số m
    while True:
        try:
            m = int(input('Nhap thong so m cho bo loc: '))
            while m not in range(1, 256):
                m = int(input('Gia tri m sai dieu kien, hay nhap lai: '))
            break
        except:
            print('Gia tri nhap vao khong hop le')
    # Chỉ cho nhập một số n
    while True:
        try:
            n = int(input('Nhap thong so n cho bo loc: '))
            while n not in range(1, 256):
                n = int(input('Gia tri n sai dieu kien, hay nhap lai: '))
            break
        except:
            print('Gia tri nhap vao khong hop le')

    result_bil = bilateral(img, d, m, n)

    return result_bil


if __name__ == '__main__':
    #Tạo một thư mục để chứa ảnh sau xử lý nếu chưa có
    if not os.path.exists(DIR_PATH):
        os.mkdir(DIR_PATH)

    PATH1 = r'C:\Users\Hieu Le\Desktop\ImageProcess\thetrue.jpg' #Tuy may ma PATH khac nhau
    PATH2 = './anhdaxuly/newimg.jpg'
    ori_img = cv2.imread(PATH1, cv2.COLOR_BGR2GRAY)

    while True:

        if not len(os.listdir('./anhdaxuly')) == 0:
            image = cv2.imread('./anhdaxuly/newimg.jpg', cv2.COLOR_BGR2GRAY)

        menu()
        n = readOnlyNumberMenu()

        if n == 1:
            print('-----------------------------------')
            print('Loc nhieu bang bo loc nao?'
                  '\n- Gaussian bam phim 1'
                  '\n- Median bam phim 2'
                  '\n- Bilateral bam phim 3')
            print('-----------------------------------')

            # Chon phuong an loc nhieu
            pa = readOnlyNumber()

            if pa == 1:  # Gaussian
                result_gaussian = locNhieu1(ori_img)
                #cv2.imwrite(os.path.join(DIR_PATH, 'newimgGAU.jpg'), result_gaussian)
                cv2.imwrite('./anhdaxuly/newimg.jpg', result_gaussian)
                showAnh(ori_img, result_gaussian)

            elif pa == 2:  # Median
                result_median = locNhieu2(ori_img)
                #cv2.imwrite(os.path.join(DIR_PATH, 'newimg.jpg'), result_median)
                cv2.imwrite('./anhdaxuly/newimg.jpg', result_median)
                showAnh(ori_img, result_median)

            elif pa == 3:  # Bilateral
                result_bilateral = locNhieu3(ori_img)
                #cv2.imwrite(os.path.join(DIR_PATH, 'newimg.jpg'), result_bilateral)
                cv2.imwrite('./anhdaxuly/newimg.jpg', result_bilateral)
                showAnh(ori_img, result_bilateral)

        elif n == 2: # Sắc nét
            print('-----------------------------------')
            print('Tang, giam do sac net:'
                  '\n- Tang sac net nhap gia tri duong.'
                  '\n- Giam sac net nhap gia tri am.')
            print('-----------------------------------')
            print('\nDo sac net duoc dieu chinh gia tri nhu sau:\n'
                  '              -5 <--- x ---> +5              ')
            print('-----------------------------------')

            while True:
                try:
                    x = float(input('Nhap gia tri de thay doi do sac net: '))
                    while x < -5 or x > 5:
                        x = float(input('Do sac net vuot nguong, hay nhap lai: '))
                    break
                except:
                    print('Gia tri nhap vao khong hop le')

            result_sacnet = sharpness2(image, x)

            cv2.imwrite('./anhdaxuly/newimg.jpg', result_sacnet)
            showAnh(image, result_sacnet)

        elif n == 3: # Làm mịn
            print('-----------------------------------')
            print('Lam min anh:'
                  '\n - Gia tri lam min thuoc doan [5, 9]')
            print('-----------------------------------')

            while True:
                try:
                    x = int(input('Nhap kich co ma tran loc: '))
                    while x not in range(5, 10):
                        x = int(input('Do lam min vuot nguong, hay nhap lai: '))
                    break
                except:
                    print('Gia tri nhap vao khong hop le')

            result_lammin = smoothing(image, x)

            cv2.imwrite('./anhdaxuly/newimg.jpg', result_lammin)
            showAnh(image, result_lammin)

        elif n == 4:    # Tuong phan
            print('-----------------------------------')
            print('Tang, giam do tuong phan:'
                  '\n- Tang tuong phan nhap gia tri duong.'
                  '\n- Giam tuong phan nhap gia tri am.')
            print('-----------------------------------')
            print('\nDo tuong phan duoc dieu chinh gia tri nhu sau:\n'
                  '              -1 <--- x ---> +3              ')
            print('-----------------------------------')

            while True:
                try:
                    xtp = float(input('Nhap do tuong phan: '))
                    while xtp > 3 or xtp <= -1:
                        xtp = float(input('Do tuong phan vuot nguong, hay nhap lai: '))
                    break
                except:
                    print('Gia tri nhap vao khong hop le')

            result_tuongphan = contrast_brightness(image, xtp, 0)

            cv2.imwrite('./anhdaxuly/newimg.jpg', result_tuongphan)
            showAnh(image, result_tuongphan)

        elif n == 5:
            print('-----------------------------------')
            print('Tang, giam do sang:'
                  '\n- Tang tuong phan nhap gia tri duong.'
                  '\n- Giam tuong phan nhap gia tri am.')
            print('-----------------------------------')
            print('\nDo sang duoc dieu chinh gia tri nhu sau:\n'
                  '           -100 <--- 0 ---> +100          ')
            print('-----------------------------------')

            while True:
                try:
                    xds = float(input('Nhap do sang: '))
                    while xds < -100 or xds > 100:
                        xds = float(input('Do sang vuot nguong, hay nhap lai: '))
                    break
                except:
                    print('Gia tri nhap vao khong hop le')

            result_dosang = contrast_brightness(image, 0, xds)

            cv2.imwrite('./anhdaxuly/newimg.jpg', result_dosang)
            showAnh(image, result_dosang)

        elif n == 6: # Tự động lọc ảnh
            print('-----------------------------------')
            print('Tu dong chinh sua anh:')
            print('Hay lua chon bo loc nhieu:'
                  '\n- Bo loc Gaussian: Phim 1'
                  '\n- Bo loc Median: Phim 2'
                  '\n- Bo loc Bilateral: Phim 3')
            print('-----------------------------------')

            choose = readOnlyNumber()

            if choose == 1:  # Bo loc Gaussian
                img_gaussian = gaussian(ori_img, 9, 9, 2)  # 9 9 = 80 210

                img_tangsacnet = sharpness2(img_gaussian, 2)

                img_lammin = smoothing(img_tangsacnet, 3)

                # edges = cv2.Canny(img_lammin, 150, 300)

                cv2.imwrite('./anhdaxuly/newimg.jpg', img_lammin)
                showAnh(ori_img, img_lammin)

            elif choose == 2:  # Bo loc Median

                img_median = median(ori_img, 7)
                img_tangsacnet = sharpness2(img_median, 2)

                img_lammin = smoothing(img_tangsacnet, 3)

                #edges = cv2.Canny(img_lammin, 100, 250)

                cv2.imwrite('./anhdaxuly/newimg.jpg', img_lammin)
                showAnh(ori_img, img_lammin)

            elif choose == 3:  # Bo loc Bilateral
                img_bilateral = bilateral(ori_img, 9, 255, 255)

                img_tangsacnet = sharpness2(img_bilateral, 2)

                img_lammin = smoothing(img_tangsacnet, 3)

                #edges = cv2.Canny(img_lammin, 100, 200)

                cv2.imwrite('./anhdaxuly/newimg.jpg', img_lammin)
                showAnh(ori_img, img_lammin)

        elif n == 7:    # Tìm biên
            print('-----------------------------------')
            print('Xac dinh duong bien cua anh')
            print('-----------------------------------')

            # Kiểm tra thư mục có rỗng không
            dir = os.listdir('./anhdaxuly')
            if len(dir) == 0:
                print('Thu muc trong, can phai thong qua cac buoc tren de co du lieu'
                      '\n..................Tu dong quay lai sau 3s......./..........')
                time.sleep(3)
            else:
                #list_img = os.listdir('./anhdaxuly') # Tạo một list chứa các tên ảnh

                # print('Ban muon tim duong bien cua anh nao?')
                # print(list_img)
                # print('-----------------------------------')
                #
                # #Nhập tên ảnh cần tìm biên vào và tạo ra đường dẫn mới để mở ảnh
                # while True:
                #     try:
                #         # B1: NHẬP
                #         img_name = input('Nhap ten anh vao day: ')
                #         while img_name not in list_img:
                #             img_name = input('Ten anh khong ton tai, hay nhap lai: ')
                #         break
                #     except:
                #         print('Gia tri khong hop le')
                #
                # # B2: TẠO ĐƯỜNG DẪN MỚI
                # new_Path = './anhdaxuly/newimg.jpg' + '/' + img_name
                #
                # # B3: MỞ ĐƯỜNG DẪN MỚI
                #EC_img = cv2.imread('./anhdaxuly/newimg.jpg', cv2.COLOR_BGR2GRAY)

                print('-----------------------------------')
                print('Nhap hai gia tri xmin va xmax'
                      '\n- xmin, xmax thuoc doan [50, 350]')
                print('-----------------------------------')

                #Nhap nguong thap
                while True:
                    try:
                        xmin = int(input('Nhap xmin: '))
                        while xmin not in range(50, 351):
                            xmin = int(input('Gia tri xmin vuot nguong, hay nhap lai: '))
                        break
                    except:
                        print('Gia tri khong hop le')

                #Nhap nguong cao
                while True:
                    try:
                        xmax = int(input('Nhap xmax: '))
                        while xmax not in range(50, 351):
                            xmax = int(input('Gia tri ymax khong vuot nguong, hay nhap lai: '))
                        break
                    except:
                        print('Gia tri khong hop le')

                result_edge = timEdge(image, xmin, xmax)

                cv2.imwrite(os.path.join('./anhdaxuly/edge.jpg'), result_edge)

                showAnh(image, result_edge)

        elif n == 8:

            while True:
                try:
                    scaleFactor = float(input('Nhap scaleFactor: '))
                    while scaleFactor < 0:
                        scaleFactor = int(input('Gia tri phai duong, hay nhap lai: '))
                    break
                except:
                    print('Gia tri khong hop le')

            while True:
                try:
                    minNeighbors = int(input('Nhap minNeighbors: '))
                    while minNeighbors < 0:
                        minNeighbors = int(input('Gia tri phai duong, hay nhap lai: '))
                    break
                except:
                    print('Gia tri khong hop le')

            face_detection.face_detection(image, scaleFactor, minNeighbors)

        elif n == 9:

            while True:
                try:
                    scaleFactor = float(input('Nhap scaleFactor: '))
                    while scaleFactor < 0:
                        scaleFactor = int(input('Gia tri phai duong, hay nhap lai: '))
                    break
                except:
                    print('Gia tri khong hop le')

            while True:
                try:
                    minNeighbors = int(input('Nhap minNeighbors: '))
                    while minNeighbors < 0:
                        minNeighbors = int(input('Gia tri phai duong, hay nhap lai: '))
                    break
                except:
                    print('Gia tri khong hop le')

            face_recognition(image, scaleFactor, minNeighbors)

        elif n == 10:
            break
