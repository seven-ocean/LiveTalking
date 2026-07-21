import os
import time
import cv2
from torch.utils.model_zoo import load_url

from ..core import FaceDetector

from .net_s3fd import s3fd
from .bbox import *
from .detect import *
from config import DEBUG_WAV2LIP_FACE_DETECTION

models_urls = {
    's3fd': 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
}


class SFDDetector(FaceDetector):
    def __init__(self, device, path_to_detector=os.path.join(os.path.dirname(os.path.abspath(__file__)), 's3fd.pth'), verbose=False):
        super(SFDDetector, self).__init__(device, verbose)

        # Initialise the face detector
        if not os.path.isfile(path_to_detector):
            model_weights = load_url(models_urls['s3fd'])
        else:
            model_weights = torch.load(path_to_detector)

        self.face_detector = s3fd()
        self.face_detector.load_state_dict(model_weights)
        self.face_detector.to(device)
        self.face_detector.eval()

    def detect_from_image(self, tensor_or_path):
        _t = time.perf_counter()
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = detect(self.face_detector, image, device=self.device)
        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] > 0.5]

        if DEBUG_WAV2LIP_FACE_DETECTION:
            print(f"  [SFDDetector] detect_from_image: {len(bboxlist)} faces, time={time.perf_counter() - _t:.3f}s")
        return bboxlist

    def detect_from_batch(self, images):
        _t = time.perf_counter()
        batch_size = images.shape[0]
        if DEBUG_WAV2LIP_FACE_DETECTION:
            print(f"  [SFDDetector] detect_from_batch: batch_size={batch_size}, image_size={images.shape[1]}x{images.shape[2]}")
            print(f"  [SFDDetector] -> calling batch_detect ...")

        bboxlists = batch_detect(self.face_detector, images, device=self.device)

        if DEBUG_WAV2LIP_FACE_DETECTION:
            print(f"  [SFDDetector] batch_detect returned, bboxlists.shape={bboxlists.shape}; running NMS ...")

        keeps = [nms(bboxlists[:, i, :], 0.3) for i in range(bboxlists.shape[1])]
        bboxlists = [bboxlists[keep, i, :] for i, keep in enumerate(keeps)]
        bboxlists = [[x for x in bboxlist if x[-1] > 0.5] for bboxlist in bboxlists]

        if DEBUG_WAV2LIP_FACE_DETECTION:
            face_counts = [len(b) for b in bboxlists]
            print(f"  [SFDDetector] detect_from_batch done: faces_per_image={face_counts}, total_time={time.perf_counter() - _t:.3f}s")
        return bboxlists

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
