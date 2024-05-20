from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class Landcover(BaseSegDataset):
    METAINFO = dict(
        classes=("background", "water", "tree", "roads", "bldg"),
        palette=[
            [0, 0, 0],
            [64, 64, 64],
            [128, 128, 128],
            [192, 192, 192],
            [255, 255, 255],
        ],
    )

    def __init__(self, **kwargs):
        super().__init__(img_suffix=".tif", seg_map_suffix=".tif", **kwargs)
