# for quantized model inference
TRAINING_PARAMS = {
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": [
            [[3, 3], [5, 6], [10, 8]],
            [[7, 15], [18, 14], [16, 32]],
            [[38, 29], [42, 67], [100, 108]],
        ],
        "classes": 6,
    },
    "batch_size": 16,
    "confidence_threshold": 0.015,
    "images_path": "./images/",
    "classes_names_path": "./data/coco_6cls.names",
    "img_h": 224,
    "img_w": 224,
    "parallels": [0],
    "pretrain_snapshot": "./weights/best_singlescale.pt",
}

# multi confidence threshold
TRAINING_PARAMS_2 = {
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": [
            [[3, 3], [5, 6], [10, 8]],
            [[7, 15], [18, 14], [16, 32]],
            [[38, 29], [42, 67], [100, 108]],
        ],
        "classes": 6,
    },
    "batch_size": 16,
    "confidence_threshold": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.1},
    "images_path": "./images/",
    "classes_names_path": "./data/coco_6cls.names",
    "img_h": 224,
    "img_w": 224,
    "parallels": [0],
    "pretrain_snapshot": "./weights/best_singlescale.pt",
}
