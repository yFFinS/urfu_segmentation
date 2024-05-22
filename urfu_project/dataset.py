from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


DATASET_COLORMAP = dict(
    background=(0, 0, 0),
    water=(64, 64, 64),
    tree=(128, 128, 128),
    roads=(192, 192, 192),
    buildings=(255, 255, 255)
    )


@DATASETS.register_module()
class Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=list(DATASET_COLORMAP.keys()),
        palette=list(DATASET_COLORMAP.values()),
    )

    def __init__(self, **kwargs):
        super().__init__(img_suffix=".tif", seg_map_suffix=".png", **kwargs)
