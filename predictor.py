import os
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer


class LeafPredictor:
    _metadata = None
    _predictor = None
    _outputs = None
    _original_img = None
    _predicted_img = None
    _img_dir = "static/imgs/"
    _img_paths = []
    _pred_boxes = None
    _pred_masks = None
    _pred_classes = None

    @property
    def predicted_img(self):
        return self._predicted_img

    @property
    def img_paths(self):
        return self._img_paths

    def __init__(self):
        # データセットを登録
        register_coco_instances(
            "leaf", {}, "PumpkinLeaf\PumpkinLeaf.json", "PumpkinLeaf/")
        self._metadata = MetadataCatalog.get("leaf")
        setup_logger()

        # 設定を決める
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation\\mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1クラスのみ
        cfg.MODEL.WEIGHTS = os.path.join(
            cfg.OUTPUT_DIR, "C:\\Users\\wakanao\\projects\\detectron2-windows\\model\\rgb_segment_model.pth")  # 絶対パスでなければならないっぽい
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
        cfg.MODEL.DEVICE = "cpu"

        # 予測器を作成
        self._predictor = DefaultPredictor(cfg)

    def predict(self, img):
        self._original_img = img
        self._outputs = self._predictor(img)
        fields = self._outputs['instances'].get_fields()
        self._pred_boxes = fields['pred_boxes']
        self._pred_masks = fields['pred_masks']
        self._pred_classes = fields['pred_classes']

        v = Visualizer(img[:, :, ::-1],
                       metadata=self._metadata,
                       scale=1.0
                       )
        v = v.draw_instance_predictions(self._outputs["instances"].to("cpu"))
        self._predicted_img = v.get_image()[:, :, ::-1]
        return self._outputs
    
    def createTrimedImg(self):
        self._img_paths = []
        boxes = self._outputs["instances"].pred_boxes
        tensor = boxes.tensor
        number_box = boxes.tensor.shape[0]

        for i in range(number_box):
            box = tensor[i]
            x1 = round(box[0].item())-5  # 画像の表示領域を10px広くする
            y1 = round(box[1].item())-5
            x2 = round(box[2].item())+5
            y2 = round(box[3].item())+5
            cut_img = self._original_img[y1:y2, x1:x2]
            img_path = self._img_dir + str(i) + ".jpg"
            self._img_paths.append(img_path)
            cv2.imwrite(img_path, cut_img)
