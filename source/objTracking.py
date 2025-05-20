import cv2
from Detector import detect
from KalmanFilter import KalmanFilter
def main():
    VideoCap = cv2.VideoCapture('cython/assests/videoplayback.mp4')

  
    ControlSpeedVar = 100 
    HiSpeed = 100

  
    #KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

    KF = KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)

    debugMode=1

    while(True):
       
        ret, frame = VideoCap.read()
        centers = detect(frame,debugMode)
        if (len(centers) > 0):
            cv2.circle(frame, (int(centers[0][0]), int(centers[0][1])), 10, (0, 191, 255), 2)
            (x, y) = KF.predict()
            cv2.rectangle(frame, (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), (255, 0, 0), 2)
            (x1, y1) = KF.update(centers[0])
            cv2.rectangle(frame, (int(x1 - 15), int(y1 - 15)), (int(x1 + 15), int(y1 + 15)), (0, 0, 255), 2)
            cv2.putText(frame, "Estimated Position", (int(x1 + 15), int(y1 + 10)), 0, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Predicted Position", (int(x + 15), int(y)), 0, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Measured Position", (int(centers[0][0] + 15), int(centers[0][1] - 15)), 0, 0.5, (0,191,255), 2)

        cv2.imshow('image', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            VideoCap.release()
            cv2.destroyAllWindows()
            break

        cv2.waitKey(HiSpeed-ControlSpeedVar+1)


if __name__ == "__main__":
    main()
