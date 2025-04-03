import torch
import backend

import onnxruntime as ort

from mmcv.parallel import MMDataParallel
from mmdet3d.models import build_model


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class BackendDeploy(backend.Backend):
    def __init__(self):
        super(BackendDeploy, self).__init__()

    def version(self):
        return torch.__version__

    def name(self):
        return "onnx-SUT"

    def image_format(self):
        return "NCHW"

    def load(self):
        return self

    def predict(self, input):
        return [0]
