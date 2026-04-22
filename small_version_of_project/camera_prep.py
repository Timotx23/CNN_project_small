from interfaces import ICameraPreper
import cv2
import platform
from cv2_enumerate_cameras import enumerate_cameras

class CameraPreper(ICameraPreper):
    def __init__(self):
        self.system = platform.system()
        self.backend = self._get_os()
        self.camera_type, self.path, self.cv_backend = self.get_camera_path()

    def _get_os(self) -> int:
        """
        Detect OS-specific OpenCV backend.
        """
        if self.system == "Darwin":
            return cv2.CAP_AVFOUNDATION
        elif self.system == "Windows":
            return cv2.CAP_MSMF
        else:
            return cv2.CAP_V4L2

    def get_camera_path(self) -> (str, int, bool):
        """
        Returns:
            ("picamera2", 0, None) for Raspberry Pi CSI camera
            ("cv2", cam_index, cam_backend) for regular webcams
        """

        # Raspberry Pi CSI camera path
        if self.system == "Linux":
            try:
                from picamera2 import Picamera2

                picam2 = Picamera2(0)
                picam2.configure(
                    picam2.create_preview_configuration(main={"size": (640, 480)})
                )
                picam2.start()
                frame = picam2.capture_array()
                picam2.stop()
                picam2.close()
                if frame is not None:
                    return "picamera2", 0, None

            except Exception as e:
                print(f"Picamera2 test failed: {e}")

        # Fallback for normal OpenCV webcams
        cams = enumerate_cameras(self.backend)

        for cam in cams:
            cam_name = cam.name.lower()

            if "facecam" in cam_name or "webcam" in cam_name or "camera" in cam_name:
                test_cap = cv2.VideoCapture(cam.index, cam.backend)

                if not test_cap.isOpened():
                    print(f"Warning: Index {cam.index} matched but failed to open.")
                success, _ = test_cap.read()
                test_cap.release()

                if  not success:
                    print(f"Warning: Index {cam.index} matched name but failed to read a frame.")
                print("CAMERA WORKING", cam.name, cam.index)
                return "cv2", cam.index, cam.backend
                    

        return False

    def open_camera(self) -> cv2.VideoCapture:
        """
        Open camera based on detected backend.
        """
        if self.camera_type == "picamera2":
            from picamera2 import Picamera2

            cam = Picamera2(self.path)
            cam.configure(
                cam.create_preview_configuration(main={"size": (640, 480)})
            )
            cam.start()
            return cam

        elif self.camera_type == "cv2":
            cam = cv2.VideoCapture(self.path, self.cv_backend)
            if not cam.isOpened():
                raise ValueError("Failed to open OpenCV camera.")
            return cam

        return False