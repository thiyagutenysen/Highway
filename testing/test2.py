import cv2

# out = cv2.VideoWriter(
#     "result/dqn____25.24mean_return__04-05-22_23-46-21/video0.mp4",
#     cv2.VideoWriter_fourcc(*"mp4v"),
#     5,
#     (600, 150),
# )
cap = cv2.VideoCapture("result/dqn____25.24mean_return__04-05-22_23-46-21/video1.mp4")
# Define the codec and create VideoWriterobject
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(
    "result/dqn____25.24mean_return__04-05-22_23-46-21/output.avi",
    fourcc,
    300.0,
    (600, 150),
)
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame, 0)

        # write the flipped frame
        out.write(frame)

        # cv2.imshow("frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
