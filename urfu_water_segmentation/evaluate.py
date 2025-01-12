import argparse
import math
import os
import pandas as pd
import cv2

from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
 
from mmseg.apis import MMSegInferencer

WATER_PIXEL_VALUE = 64
BACKGROUND_CLASS_IDX = 0
WATER_CLASS_IDX = 1

PALETTE = [
    [0, 0, 0],
    [64, 64, 64],
]

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
    parser.add_argument("-o", "--output-path")
    parser.add_argument("-V", "--visualize", required=False, default=False, action="store_true")
    parser.add_argument("-e", "--experiment-path")
    parser.add_argument("--device", required=False, default=None)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True)
    
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
        dataset_name = dataset['name']
        images_path, gt_path = dataset['images'], dataset['gt']
        
        predictions = []
        images = [str(path) for path in images_path.glob("*.tif")]
        
        print(f'Inferencing {dataset_name}. Images {len(images)}')
        
        for i in range(math.ceil(len(images) / batch_size)):
            inference_result = inferencer(
                images[i * batch_size: (i + 1) * batch_size],
                batch_size=batch_size,
            )
            predictions.extend(inference_result['predictions'])
        
        print(f'Scoring {dataset_name}')
        
        gts = []
        for gt_image in gt_path.glob("*.tif"):
            gt = cv2.imread(str(gt_image), cv2.IMREAD_GRAYSCALE)
            
            gt[gt != WATER_PIXEL_VALUE] = BACKGROUND_CLASS_IDX
            gt[gt == WATER_PIXEL_VALUE] = WATER_CLASS_IDX
            
            gts.append(gt)
        
        acc_values = []
        iou_values = []
        
        for pred, gt in zip(predictions, gts):
            iou, acc = calculate_iou_and_accuracy(pred, mask=gt)
            
            iou_values.append(iou)
            acc_values.append(acc)
        
        if args.visualize:
            print(f'Visualizing {dataset_name}')
            for idx, (image_path, pred, gt) in enumerate(zip(images, predictions, gts)):
                image = Image.open(image_path)
                pred_image = Image.fromarray(pred.astype(np.uint8)).convert('P')
                pred_image.putpalette(np.array(PALETTE, dtype=np.uint8))
                
                plot_name = os.path.basename(image_path) + f"_iou_{iou_values[idx]:.2f}_acc_{acc_values[idx]:.2f}.png"
                save_path = output_path / 'visualization' / dataset_name / plot_name
                save_path.mkdir(parents=True, exists_ok=True)
                
                plots = {
                    'image': image,
                    'predicted': np.array(pred_image),
                    'mask': gt,
                }
                write_plots_and_visualize(
                    str(save_path),
                    **plots,
                )
            
        results[dataset_name] = {
            'mAcc': np.mean(acc_values),
            'mIoU': np.mean(iou_values),
        }
        print(f'{dataset_name} results: {results[dataset_name]}')
            
    output_df = pd.DataFrame.from_dict(results, orient='index')
    print(output_df)
    output_df.to_csv(output_path / 'results.csv')


def calculate_iou_and_accuracy(predicted, mask):
    eps = 1e-8
    
    intersect = predicted[predicted == mask]
    area_intersect, _ = np.histogram(intersect, bins=2, range=(0, 1))
    area_pred, _ = np.histogram(predicted, bins=2, range=(0, 1))
    area_gt, _ = np.histogram(mask, bins=2, range=(0, 1))
    area_union = area_pred + area_gt - area_intersect
    
    iou = (area_intersect + eps) / (area_union + eps)
    acc = (area_intersect + eps) / (area_gt + eps)
    
    return iou, acc

def write_plots_and_visualize(path_to_res, **images):
    """Функция для визуализации данных, располагает изображения в ряд"""
    n = len(images)
    plt.figure(figsize=(16, 5))

    for i, (name, image) in enumerate(images.items()):
        if image is not None:
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)

    plt.savefig(path_to_res, bbox_inches='tight')
        
if __name__ == "__main__":
    main()
