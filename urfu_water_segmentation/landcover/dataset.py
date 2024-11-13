from collections import defaultdict
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


DATASET_COLORMAP = dict(
    background=(0, 0, 0),
    water=(64, 64, 64),
    tree=(128, 128, 128),
    roads=(192, 192, 192),
    buildings=(255, 255, 255),
)

DATASET_CLASS_MAPPING = {
    'tree': 'background',
    'roads': 'background',
    'buildings': 'background',
}


@DATASETS.register_module()
class LandcoverAI(BaseSegDataset):
    METAINFO = dict(
        classes=list(DATASET_COLORMAP.keys()),
        palette=list(DATASET_COLORMAP.values()),
    )

    def __init__(self, **kwargs):
        new_classes = []
        self._data_label_map = {}
        
        for key, value in DATASET_COLORMAP.items():
            new_key = DATASET_CLASS_MAPPING.get(key, key)
            
            if new_key not in new_classes:
                new_classes.append(new_key)
            
            self._data_label_map[value[0]] = new_classes.index(new_key)
            
        super().__init__(
            img_suffix=".tif",
            seg_map_suffix=".tif",
            metainfo={'classes': new_classes},
            **kwargs
        )
        
    def get_data_info(self, idx):
        result = super().get_data_info(idx)
        if result is None:
            return None
        
        result['label_map'] = self._data_label_map.copy()
        return result