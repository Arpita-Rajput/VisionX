import cv2
from camera.camera_manager import CameraManager


camera = CameraManager(0)

while True:

    frame = camera.read_frame()

    if frame is None:
        print("Unable to read camera")
        break

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()