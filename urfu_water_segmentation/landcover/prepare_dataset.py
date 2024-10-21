import numpy as np
from PIL import Image
from tqdm import tqdm

from pathlib import Path

data_root = Path('/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512')

def add_palette_for_1ch_gt(input_path: Path, output_path: Path,
                           palette=(
                                (0, 0, 0),  # 0
                                (64, 64, 64),  # 1
                                (128, 128, 128),  # 2
                                (192, 192, 192),  # 3
                                (255, 255, 255),  # 4
                            ),
):
    # Select only water
    palette_dict = {int(color[0]): int(color[0] == 64) for i, color in enumerate(palette)}
    map_fn = np.vectorize(palette_dict.get)
    
    for img_path in tqdm(list(input_path.glob('*.tif'))):
        im = Image.open(img_path)
        ar = map_fn(np.array(im)).astype(np.uint8)
        im = Image.fromarray(ar).convert('P')
        im.putpalette(np.array(palette, dtype=np.uint8))
        
        new_image_name = img_path.name.split('.')[0] + '.png'
        im.save(output_path / new_image_name)


def symlink_content(input_path: Path, output_path: Path):
    for path in tqdm(list(input_path.glob('*'))):
        output_image = output_path / path.name
        if not output_image.exists():
            output_image.symlink_to(path, target_is_directory=False)


def main():
    train_data = data_root / 'train'
    val_data = data_root / 'val'
    output_images = data_root / 'images'
    output_gts = data_root / 'gt'
    
    output_images.mkdir(exist_ok=True)
    output_gts.mkdir(exist_ok=True)
    
    print("Creating symlinks...")
    symlink_content(train_data / 'images', output_images)
    symlink_content(val_data / 'images', output_images)
    
    print("Converting palettes...")
    add_palette_for_1ch_gt(train_data / 'gt', output_gts)
    add_palette_for_1ch_gt(val_data / 'gt', output_gts)    

    
if __name__ == "__main__":
    main()