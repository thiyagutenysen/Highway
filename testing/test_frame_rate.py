import cv2

cap = cv2.VideoCapture("result/dqn____25.24mean_return__04-05-22_23-46-21/video0.mp4")
cap.set(cv2.CAP_PROP_FPS, 1000)
# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame
        cv2.imshow("Frame", frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
