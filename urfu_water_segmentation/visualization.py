import argparse
from pathlib import Path
import shutil
import mmcv
import os
from PIL import Image

import matplotlib.pyplot as plt
from mmengine.structures import PixelData
import numpy as np

# `SegDataSample` is data structure interface between different components
# defined in MMSegmentation, it includes ground truth, prediction and
# predicted logits of semantic segmentation.
# Please refer to below tutorial file of `SegDataSample` for more details:
# https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/en/advanced_guides/structures.md

from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer
from mmseg.apis.mmseg_inferencer import MMSegInferencer

CLASS_NAMES = ['background', 'water']
PALETTE = [
    [0, 0, 0],
    [64, 64, 64],
]

def find_config_path(experiment_path: Path) -> Path:
    return next(experiment_path.glob("*.py"))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment-path', required=True)
    parser.add_argument('-i', '--images-path', required=True)
    parser.add_argument('-m', '--masks-path', required=False, default=None)
    parser.add_argument('-o', '--output-path', required=True)
    parser.add_argument('-f', '--overwrite', action='store_true', required=False, default=False)
    parser.add_argument('--device', required=False, default='cpu')
    parser.add_argument('--batch-size', required=False, default=1)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    images_path = Path(args.images_path)
    masks_path = Path(args.masks_path) if args.masks_path else None
    output_path = Path(args.output_path)
    batch_size = int(args.batch_size)
    
    if not images_path.exists():
        raise RuntimeError("Input does not exist")
    
    if output_path.exists():
        if not args.overwrite:
            raise RuntimeError("Output already exists. Pass -f to overwrite")
        shutil.rmtree(output_path)
    
    experiment_path = Path(args.experiment_path)
    
    config_path = str(find_config_path(experiment_path))
    weights_path = (experiment_path / 'last_checkpoint').read_text()
    inferencer = MMSegInferencer(
        model=config_path,
        weights=weights_path,
        device=args.device,
    )
    
    inferencer.show_progress = True
    
    image_paths = [str(image_path) for image_path in images_path.glob("*")]
    mask_paths = [str(mask_path) for mask_path in masks_path.glob("*")] if masks_path else None
    
    image_paths = image_paths[::100]
    mask_paths = mask_paths[::100]
    
    results = inferencer(image_paths, batch_size=batch_size, return_vis=False)
    
    predictions = results['predictions']
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx, image_path in enumerate(image_paths):
        out_file = str(output_path / os.path.basename(image_path))
        # Force .png
        out_file += '.png'
        
        image = Image.open(image_path)
        
        pred_image = Image.fromarray(predictions[idx].astype(np.uint8)).convert('P')
        pred_image.putpalette(np.array(PALETTE, dtype=np.uint8))
        
        plots = {'image': image, 'predicted': np.array(pred_image)}
        if mask_paths:
            mask = Image.open(mask_paths[idx])
            mask = np.array(mask)
            # Leave only water
            mask[mask != 64] = 0
            plots['mask'] = mask
            
        write_plots_and_visualize(
            out_file,
            **plots,
        )
# if args.plots:  # сохраняем снимок/выход сети/ разметку в одну картинку в папку plots
#         if _b_calc_metrics:
#             name, ext = filename.split('.')
#             plot_name = name + f"_iou_{metrics['iou']:.2f}_acc_{metrics['acc']:.2f}." + ext
#         else:
#             plot_name = filename
#         write_plots_and_visualize(  # (path_to_res, visualize=False, **images)
#             os.path.join(_output_plots_path, plot_name),
#             image=image,
#             predicted=res['plot'],
#             mask=mask
#         )

def write_plots_and_visualize(path_to_res, visualize=False, **images):
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
    print(f'Save {path_to_res}')

    if visualize:
        plt.show()

if __name__ == "__main__":
    main()