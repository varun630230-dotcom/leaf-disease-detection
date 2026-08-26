import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
import random
import numpy as np
import mlflow
from tqdm import tqdm
import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_transforms(config):
    img_size = config['dataset']['image_size']
    aug = config['augmentation']
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=tuple(aug['random_crop_scale'])),
        transforms.RandomHorizontalFlip(p=aug['horizontal_flip']),
        transforms.RandomRotation(aug['rotation_degrees']),
        transforms.ColorJitter(
            brightness=aug['color_jitter']['brightness'],
            contrast=aug['color_jitter']['contrast'],
            saturation=aug['color_jitter']['saturation']
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model(num_classes, pretrained, dropout_rate):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    
    # Custom head
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(num_ftrs, num_classes),
    )
    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(config['training']['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_transform, val_transform = get_transforms(config)
    
    data_dir = Path("backend/data")
    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'])
    
    model = create_model(config['dataset']['num_classes'], config['model']['pretrained'], config['model']['dropout'])
    model = model.to(device)
    
    # Class weights for imbalanced dataset
    class_counts = [0] * config['dataset']['num_classes']
    for _, target in train_dataset:
        class_counts[target] += 1
    
    weights = [1.0 / (c + 1e-5) for c in class_counts]
    weights = torch.FloatTensor(weights).to(device)
    weights = weights / weights.sum() * len(weights)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    mlflow.set_experiment("leafguard_training")
    
    with mlflow.start_run():
        mlflow.log_params(config['training'])
        mlflow.log_params(config['model'])
        
        # Stage 1: Train Head
        print("Stage 1: Training classifier head...")
        for param in model.features.parameters():
            param.requires_grad = False
            
        optimizer_head = optim.Adam(model.classifier.parameters(), lr=config['training']['stage1_lr'])
        
        for epoch in range(config['training']['stage1_epochs']):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer_head, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            
            print(f"Stage 1 - Epoch {epoch+1}: Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
        # Stage 2: Fine-tune backbone
        print("Stage 2: Fine-tuning full model...")
        for param in model.parameters():
            param.requires_grad = True
            
        optimizer_full = optim.Adam([
            {'params': model.features.parameters(), 'lr': config['training']['stage2_lr_backbone']},
            {'params': model.classifier.parameters(), 'lr': config['training']['stage2_lr_head']}
        ], weight_decay=config['training']['weight_decay'])
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_full, T_max=config['training']['stage2_epochs'])
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(config['training']['stage2_epochs']):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer_full, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            
            print(f"Stage 2 - Epoch {epoch+1}: Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
            mlflow.log_metrics({
                'train_loss': train_loss, 'train_acc': train_acc,
                'val_loss': val_loss, 'val_acc': val_acc
            }, step=epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= config['training']['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        # Save Best Model
        model.load_state_dict(best_model_state)
        
        out_dir = Path("backend/models/classifier")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "best_model.pth")
        
        with open(out_dir / "preprocessing_config.json", "w") as f:
            json.dump({
                "image_size": config['dataset']['image_size'],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }, f, indent=2)
            
        with open(out_dir / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)
            
        print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
