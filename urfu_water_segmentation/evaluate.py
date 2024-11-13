import argparse
import math
import pandas as pd
import cv2

from pathlib import Path

import numpy as np
 
from mmseg.apis import MMSegInferencer

WATER_PIXEL_VALUE = 64
BACKGROUND_CLASS_IDX = 0
WATER_CLASS_IDX = 1

landcover_path = Path('/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512')
glh_water_path = Path('/misc/home1/m_imm_freedata/Segmentation/Projects/GLH_water/glh_cut_512_filtered')
deepglobe_path = Path('/misc/home1/m_imm_freedata/Segmentation/DeepGlobe_Land/DeepGlobe512')
loveda_path = Path('/misc/home1/m_imm_freedata/Segmentation/LoveDA')
rg3_path = Path('/misc/home1/m_imm_freedata/Segmentation/RG3/train_and_val')

batch_size = 32

# Для каждого датасета будет посчитаны свои метрики
# Для датасетов нужно указать
# - name - название датасета
# - images - папка с изображениями из датасета, будут использованы все 
# - gt - папка с разметкой из датасета
# ! Порядок изображений в images и в gt должен совпадать, а так же ожидается, что все изображения формата .tif
datasets = [
    {'name': 'landcover', 'images': landcover_path / 'val' / 'images', 'gt': landcover_path / 'val' / 'gt'},
    {'name': 'glh_water', 'images': glh_water_path / 'val' / 'images', 'gt': glh_water_path / 'val' / 'gt'},
    {'name': 'deepglobe', 'images': deepglobe_path / 'val' / 'images', 'gt': deepglobe_path / 'val' / 'gt'},
    {'name': 'loveda', 'images': loveda_path / 'val' / 'images', 'gt': loveda_path / 'val' / 'gt'},
    {'name': 'rg3', 'images': rg3_path / 'images', 'gt': rg3_path / 'gt'},
]

def find_config_path(experiment_path: Path) -> Path:
    return next(experiment_path.glob("*.py"))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path")
    parser.add_argument("--experiment-path")
    parser.add_argument("--device", required=False, default=None)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    experiment_path = Path(args.experiment_path)
    
    config_path = str(find_config_path(experiment_path))
    weights_path = (experiment_path / 'last_checkpoint').read_text()
    inferencer = MMSegInferencer(
        model=config_path,
        weights=weights_path,
        device=args.device,
    )
    inferencer.show_progress = False
    
    results = {}
    
    for dataset in datasets:
        name = dataset['name']
        images_path, gt_path = dataset['images'], dataset['gt']
        
        predictions = []
        images = [str(path) for path in images_path.glob("*.tif")]
        
        print(f'Inferencing {name}. Images {len(images)}')
        
        for i in range(math.ceil(len(images) / batch_size)):
            inference_result = inferencer(
                images[i * batch_size: (i + 1) * batch_size],
                batch_size=batch_size,
            )
            predictions.extend(inference_result['predictions'])
        
        print(f'Scoring {name}')
        
        gts = []
        for gt_image in gt_path.glob("*.tif"):
            gt = cv2.imread(str(gt_image), cv2.IMREAD_GRAYSCALE)
            
            gt[gt != WATER_PIXEL_VALUE] = BACKGROUND_CLASS_IDX
            gt[gt == WATER_PIXEL_VALUE] = WATER_CLASS_IDX
            
            gts.append(gt)
        
        acc_values = []
        iou_values = []
        eps = 1e-8
        
        for pred, gt in zip(predictions, gts):
            intersect = pred[pred == gt]
            area_intersect, _ = np.histogram(intersect, bins=2, range=(0, 1))
            area_pred, _ = np.histogram(pred, bins=2, range=(0, 1))
            area_gt, _ = np.histogram(gt, bins=2, range=(0, 1))
            area_union = area_pred + area_gt - area_intersect
            
            iou = (area_intersect + eps) / (area_union + eps)
            acc = (area_intersect + eps) / (area_gt + eps)
            
            iou_values.append(iou)
            acc_values.append(acc)
            
        results[name] = {
            'mAcc': np.mean(acc_values),
            'mIoU': np.mean(iou_values),
        }
        print(f'{name} results: {results[name]}')
            
    output_df = pd.DataFrame.from_dict(results, orient='index')
    print(output_df)
    output_df.to_csv(args.output_path)


if __name__ == "__main__":
    main()
