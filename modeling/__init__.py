from .config import add_boundaryformer_config
from .new_rcnn import NewGeneralizedRCNN
from .roi_heads import *
from .postprocessing import *
from .soft_polygon import SoftPolygonBatch
from . import data # register new data set
from .criterion import MaskCriterion
from .backbone.swin import D2SwinTransformer, build_swin_fpn_backbone
from .backbone.fcos_backbone import build_fcos_resnet_fpn_backbone
from .fcos import FCOS
