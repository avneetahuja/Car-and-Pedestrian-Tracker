import cv2 as cv

# img = "Cars6.png"
vid = cv.VideoCapture('video3.mp4')
# key = cv.waitKey(1)
trainedCarFile = "car_detector.xml"

# trainedPedFile = "ped.xml"

while True:
    succesfulRead, frame = vid.read()
    
    # img = cv.imread(img)

    # pedTracker = cv.CascadeClassifier(trainedPedFile)

    carTracker = cv.CascadeClassifier(trainedCarFile)

    grayImg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # peds = pedTracker.detectMultiScale(grayImg)#,1.05,1,0,[60,60])

    cars = carTracker.detectMultiScale(grayImg,1.1,2,0,[70,70])

    for(x,y,w,h) in cars:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    # for (x,y,w,h) in peds:
        # cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    cv.imshow("CTApp",frame)

    key = cv.waitKey(1)
    
    if key==81 or key==113:
        break
    
vid.release()


print("Code Run Done")
