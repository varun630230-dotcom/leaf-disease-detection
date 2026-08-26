import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import json
import yaml
from .train_classifier import create_model
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)
        
    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

def set_temperature(valid_loader, model, device, config):
    nll_criterion = nn.CrossEntropyLoss().to(device)
    ece_criterion = ECELoss().to(device)

    model_with_temp = ModelWithTemperature(model).to(device)
    
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for input, label in valid_loader:
            input = input.to(device)
            logits = model(input)
            logits_list.append(logits)
            labels_list.append(label)
            
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

    # ECE before calibration
    before_temperature_nll = nll_criterion(logits, labels).item()
    before_temperature_ece = ece_criterion(logits, labels).item()
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
    
    optimizer = torch.optim.LBFGS([model_with_temp.temperature], lr=config['calibration']['temperature_lr'], max_iter=config['calibration']['temperature_max_iter'])

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(model_with_temp.temperature_scale(logits), labels)
        loss.backward()
        return loss
        
    optimizer.step(eval)

    # ECE after calibration
    after_temperature_nll = nll_criterion(model_with_temp.temperature_scale(logits), labels).item()
    after_temperature_ece = ece_criterion(model_with_temp.temperature_scale(logits), labels).item()
    print('Optimal temperature: %.3f' % model_with_temp.temperature.item())
    print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
    
    return model_with_temp.temperature.item(), before_temperature_ece, after_temperature_ece

class ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

def calculate_energy(logits, temperature=1.0):
    return -temperature * torch.logsumexp(logits / temperature, dim=1)

def calibrate_ood(model, id_loader, ood_loader, device, config):
    model.eval()
    temp = config['ood']['energy_temperature']
    
    id_energies = []
    with torch.no_grad():
        for inputs, _ in tqdm(id_loader, desc="ID Energies"):
            inputs = inputs.to(device)
            logits = model(inputs)
            energy = calculate_energy(logits, temp)
            id_energies.extend(energy.cpu().numpy())
            
    id_energies = np.array(id_energies)
    threshold = np.percentile(id_energies, config['ood']['id_percentile'])
    print(f"OOD Energy Threshold (at {config['ood']['id_percentile']}th percentile): {threshold:.4f}")
    
    auroc = None
    fpr95 = None
    if ood_loader and len(ood_loader.dataset) > 0:
        ood_energies = []
        with torch.no_grad():
            for inputs, _ in tqdm(ood_loader, desc="OOD Energies"):
                inputs = inputs.to(device)
                logits = model(inputs)
                energy = calculate_energy(logits, temp)
                ood_energies.extend(energy.cpu().numpy())
                
        ood_energies = np.array(ood_energies)
        y_true = np.concatenate([np.zeros(len(id_energies)), np.ones(len(ood_energies))])
        y_scores = np.concatenate([id_energies, ood_energies])  # higher energy = more OOD
        
        auroc = roc_auc_score(y_true, y_scores)
        print(f"OOD Detection AUROC: {auroc:.4f}")
        
    return threshold, auroc, fpr95

def main():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_dir = Path("backend/models/classifier")
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config['dataset']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder("backend/data/val", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    ood_dataset = datasets.ImageFolder("backend/data/ood_test", transform=val_transform)
    ood_loader = DataLoader(ood_dataset, batch_size=config['training']['batch_size'], shuffle=False) if len(ood_dataset) > 0 else None
    
    model = create_model(config['dataset']['num_classes'], False, config['model']['dropout'])
    model.load_state_dict(torch.load(model_dir / "best_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    
    print("Calibrating Temperature...")
    opt_t, ece_before, ece_after = set_temperature(val_loader, model, device, config)
    
    with open(model_dir / "calibration.json", "w") as f:
        json.dump({"temperature": opt_t, "ece_before": ece_before, "ece_after": ece_after}, f, indent=2)
        
    print("Calibrating OOD...")
    threshold, auroc, fpr95 = calibrate_ood(model, val_loader, ood_loader, device, config)
    
    with open(model_dir / "ood_config.json", "w") as f:
        json.dump({
            "energy_temperature": config['ood']['energy_temperature'],
            "threshold": float(threshold),
            "auroc": float(auroc) if auroc else None,
            "fpr95": float(fpr95) if fpr95 else None
        }, f, indent=2)
        
    with open(model_dir / "confidence_config.json", "w") as f:
        json.dump(config['confidence'], f, indent=2)

if __name__ == "__main__":
    main()
