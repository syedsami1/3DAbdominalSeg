import torch
from torch.utils.data import DataLoader
from models.vnet import VNet
from dataset import CTAbdomenDataset
import yaml
from sklearn.metrics import dice_score  # Assuming you have a Dice score function

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VNet().to(device)
    model.load_state_dict(torch.load(config['paths']['model_save_path']))
    model.eval()
    
    test_dataset = CTAbdomenDataset(config['paths']['test_data'], transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    dice_scores = { 'Liver': 0, 'Right Kidney': 0, 'Left Kidney': 0, 'Spleen': 0 }
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            # Compute Dice score here
            # dice_scores = compute_dice(outputs, labels)
    
    print("Dice Scores:", dice_scores)

if __name__ == "__main__":
    main()
