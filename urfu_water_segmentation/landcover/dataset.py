from typing import Sequence
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


class ClassMapping(dict, Sequence):
    def index(self, item):
        return self[item]

DATASET_CLASSES = ClassMapping(
    background=0,
    water=1,
    tree=0,
    roads=0,
    buildings=0,
)
DATASET_COLORMAP = dict(
    background=(0, 0, 0),
    water=(64, 64, 64),
)


@DATASETS.register_module()
class LandcoverAI(BaseSegDataset):
    METAINFO = dict(
        classes=DATASET_CLASSES,
        palette=list(DATASET_COLORMAP.values()),
    )

    def __init__(self, **kwargs):
        super().__init__(img_suffix=".tif", seg_map_suffix=".tif", **kwargs)
