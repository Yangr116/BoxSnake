from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
import os


# [{"supercategory": "aeroplane", "name": "aeroplane", "id": 1}, 
# {"supercategory": "bicycle", "name": "bicycle", "id": 2}, 
# {"supercategory": "bird", "name": "bird", "id": 3}, 
# {"supercategory": "boat", "name": "boat", "id": 4}, 
# {"supercategory": "bottle", "name": "bottle", "id": 5}, 
# {"supercategory": "bus", "name": "bus", "id": 6}, 
# {"supercategory": "car", "name": "car", "id": 7}, 
# {"supercategory": "cat", "name": "cat", "id": 8}, 
# {"supercategory": "chair", "name": "chair", "id": 9}, 
# {"supercategory": "cow", "name": "cow", "id": 10}, 
# {"supercategory": "diningtable", "name": "diningtable", "id": 11},
# {"supercategory": "dog", "name": "dog", "id": 12}, 
# {"supercategory": "horse", "name": "horse", "id": 13}, 
# {"supercategory": "motorbike", "name": "motorbike", "id": 14}, 
# {"supercategory": "person", "name": "person", "id": 15}, 
# {"supercategory": "pottedplant", "name": "pottedplant", "id": 16}, 
# {"supercategory": "sheep", "name": "sheep", "id": 17}, 
# {"supercategory": "sofa", "name": "sofa", "id": 18}, 
# {"supercategory": "train", "name": "train", "id": 19}, 
# {"supercategory": "tvmonitor", "name": "tvmonitor", "id": 20}]

VOC_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "aeroplane"}, # should be airplane, since wrong in json
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "bird"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "boat"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "bottle"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "car"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "cat"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "chair"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "cow"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "diningtable"},
    {"color": [220, 220, 0], "isthing": 1, "id": 12, "name": "dog"},
    {"color": [175, 116, 175], "isthing": 1, "id": 13, "name": "horse"},
    {"color": [250, 0, 30], "isthing": 1, "id": 14, "name": "motorbike"},
    {"color": [120, 166, 157], "isthing": 1, "id": 15, "name": "person"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "pottedplant"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "sheep"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "sofa"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "train"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "tvmonitor"},
]

def _get_voc_instances_meta():
    thing_ids = [k["id"] for k in VOC_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in VOC_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 20, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in VOC_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


_PREDEFINED_SPLITS = {
    # point annotations without masks
    "voc_instance_seg_train": (
        "VOC/train",
        "VOC/annotations/voc_2012_train.json",
    ),
    "voc_instance_seg_val": (
        "VOC/val",
        "VOC/annotations/voc_2012_val.json",
    ),
}

def register_voc_instance_segmentation(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_voc_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_voc_instance_segmentation(_root)
