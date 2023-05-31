import cv2, os

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeDetector = cv2.CascadeClassifier('haarcascade_eye.xml')
faceID = input("masukan face id yang akan direkam datanya [kemudian tekan enter]: ")
print("tatap wajah anda ke depan dalam webcam. Silahkan tunguu...")
ambilData = 1
while True: 
    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255),2)
        namaFile = 'wajah'+str(faceID)+'.'+str(ambilData)+'.jpg'
        cv2.imwrite('datawajah'+'/'+namaFile, frame)
        ambilData +=1
        roiAbuAbu = abuAbu[y:y+h,x:x+w]
        roiWarna = frame[y:y+h,x:x+h]
        eyes = eyeDetector.detectMultiScale(roiAbuAbu)
        for (xe, ye, we, he) in eyes:
            cv2.rectangle(roiWarna, (xe,ye), (xe+we,ye+he), (0,0,255),1)
    cv2.imshow('webcamku', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or  k == ord('q'):
        break
    elif ambilData > 30:
        print('wajah telah selesai dipindai!')
        break
cam.release()
cv2.destroyAllWindows()