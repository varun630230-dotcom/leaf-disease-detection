import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import json
import yaml
from .train_classifier import create_model
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def measure_latency(model, device, img_size):
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    model.eval()
    
    # warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy_input)
            
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            model(dummy_input)
        end = time.perf_counter()
        latencies.append((end - start) * 1000) # ms
        
    return {
        "mean_ms": np.mean(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95)
    }

def main():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path("backend/models/classifier")
    eval_dir = Path("backend/models/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configs
    with open(model_dir / "class_mapping.json", "r") as f:
        class_mapping = json.load(f)
        
    class_names = [class_mapping[str(i)]['class_name'] for i in range(len(class_mapping))]
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config['dataset']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder("backend/data/test", transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    model = create_model(config['dataset']['num_classes'], False, config['model']['dropout'])
    model.load_state_dict(torch.load(model_dir / "best_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Classification Report
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    with open(eval_dir / "evaluation_report.json", "w") as f:
        json.dump({
            "accuracy": report_dict["accuracy"],
            "macro_avg": report_dict["macro avg"],
            "weighted_avg": report_dict["weighted avg"]
        }, f, indent=2)
        
    with open(eval_dir / "per_class_metrics.json", "w") as f:
        per_class = {k: v for k, v in report_dict.items() if k not in ["accuracy", "macro avg", "weighted avg"]}
        json.dump(per_class, f, indent=2)
        
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(eval_dir / "confusion_matrix.png")
    plt.close()
    
    # Latency
    lat_stats = measure_latency(model, device, config['dataset']['image_size'])
    model_size_mb = os.path.getsize(model_dir / "best_model.pth") / (1024 * 1024)
    lat_stats["model_size_mb"] = model_size_mb
    
    with open(eval_dir / "latency_report.json", "w") as f:
        json.dump(lat_stats, f, indent=2)
        
    # OOD Eval (Optional)
    ood_dir = Path("backend/data/ood_test")
    if ood_dir.exists() and len(list(ood_dir.glob("*/*.*"))) > 0:
        ood_dataset = datasets.ImageFolder("backend/data/ood_test", transform=val_transform)
        ood_loader = DataLoader(ood_dataset, batch_size=config['training']['batch_size'], shuffle=False)
        
        with open(model_dir / "ood_config.json", "r") as f:
            ood_config = json.load(f)
            
        temp = ood_config['energy_temperature']
        threshold = ood_config['threshold']
        
        id_energies = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                logits = model(inputs)
                energy = -temp * torch.logsumexp(logits / temp, dim=1)
                id_energies.extend(energy.cpu().numpy())
                
        ood_energies = []
        with torch.no_grad():
            for inputs, _ in ood_loader:
                inputs = inputs.to(device)
                logits = model(inputs)
                energy = -temp * torch.logsumexp(logits / temp, dim=1)
                ood_energies.extend(energy.cpu().numpy())
                
        id_energies = np.array(id_energies)
        ood_energies = np.array(ood_energies)
        
        y_true = np.concatenate([np.zeros(len(id_energies)), np.ones(len(ood_energies))])
        y_scores = np.concatenate([id_energies, ood_energies])
        
        auroc = roc_auc_score(y_true, y_scores)
        rejection_rate = np.mean(ood_energies > threshold)
        
        with open(eval_dir / "ood_metrics.json", "w") as f:
            json.dump({
                "auroc": float(auroc),
                "rejection_rate": float(rejection_rate)
            }, f, indent=2)

if __name__ == "__main__":
    main()
