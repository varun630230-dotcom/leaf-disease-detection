import json
import os
import shutil
import random
from pathlib import Path
import yaml
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

def parse_class_name(folder_name):
    """Parse 'Plant___Disease' to plant and disease strings."""
    parts = folder_name.split('___')
    plant = parts[0].replace('_', ' ')
    if len(parts) > 1:
        disease = parts[1].replace('_', ' ')
        is_healthy = 'healthy' in disease.lower()
    else:
        disease = 'Unknown'
        is_healthy = False
    return plant, disease, is_healthy

def prepare():
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    train_ratio = config['dataset']['train_ratio']
    val_ratio = config['dataset']['val_ratio']
    test_ratio = config['dataset']['test_ratio']
    
    # Download dataset
    print("Downloading PlantVillage dataset...")
    dataset = load_dataset('mohanty/PlantVillage', 'color')
    
    data_dir = Path("backend/data")
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    ood_dir = data_dir / "ood_test"
    
    for d in [train_dir, val_dir, test_dir, ood_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Read README for OOD
    with open(ood_dir / "README.md", "w") as f:
        f.write("# OOD Test Images\n\nPlease manually download Out-Of-Distribution images here for OOD evaluation.\n")
        
    # Group images
    class_names = dataset['train'].features['label'].names
    
    # To avoid data leakage, group by leaf ID (uuid prefix) if possible, but PlantVillage 
    # datasets library structure typically gives images. We will try to extract UUID from path if available.
    # Since we can't easily extract UUID from HuggingFace dataset without looking at the cache, we'll do standard stratified split.
    
    class_groups = defaultdict(list)
    for i, item in enumerate(tqdm(dataset['train'], desc="Grouping dataset")):
        class_groups[item['label']].append(item['image'])
        
    print("\nDataset Statistics:")
    total_count = sum(len(items) for items in class_groups.values())
    print(f"Total images: {total_count}")
    
    class_mapping = {}
    
    for label_idx, images in class_groups.items():
        class_name = class_names[label_idx]
        plant, disease, is_healthy = parse_class_name(class_name)
        
        class_mapping[str(label_idx)] = {
            "class_name": class_name,
            "plant": plant,
            "disease": disease,
            "is_healthy": is_healthy
        }
        
        print(f"{class_name}: {len(images)}")
        
        # Shuffle
        random.seed(config['training']['seed'])
        random.shuffle(images)
        
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]
        
        # Save images
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        (test_dir / class_name).mkdir(exist_ok=True)
        
        for i, img in enumerate(train_imgs):
            img.save(train_dir / class_name / f"{i}.jpg")
            
        for i, img in enumerate(val_imgs):
            img.save(val_dir / class_name / f"{i}.jpg")
            
        for i, img in enumerate(test_imgs):
            img.save(test_dir / class_name / f"{i}.jpg")
            
    # Save mapping
    models_dir = Path("backend/models/classifier")
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
        
    print("\nDataset preparation complete.")

if __name__ == "__main__":
    prepare()
