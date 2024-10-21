from pathlib import Path

root_dir = Path('/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512')

def main():
    train_images = (root_dir / 'train' / 'images').glob("*.tif")
    val_images = (root_dir / 'val' / 'images').glob("*.tif")

    splits_dir = root_dir / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    train_images_names = [path.name.split('.')[0] for path in train_images]
    val_images_names = [path.name.split('.')[0] for path in val_images]
    
    (splits_dir / 'train.txt').write_text('\n'.join(train_images_names))
    (splits_dir / 'val.txt').write_text('\n'.join(val_images_names))


if __name__ == "__main__":
    main()