import cv2


class CameraManager:

    def __init__(self, source=0):
        self.source = source
        self.camera = cv2.VideoCapture(source)

    def read_frame(self):
        success, frame = self.camera.read()

        if not success:
            return None

        return frame

    def release(self):
        self.camera.release()