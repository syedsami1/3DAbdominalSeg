import torch
import matplotlib.pyplot as plt
from models.vnet import VNet
from dataset import CTAbdomenDataset
import yaml

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VNet().to(device)
    model.load_state_dict(torch.load(config['paths']['model_save_path']))
    model.eval()
    
    test_dataset = CTAbdomenDataset(config['paths']['test_data'], transform=transforms.ToTensor())
    
    with torch.no_grad():
        for i, data in enumerate(test_dataset):
            inputs, _ = data
            inputs = inputs.unsqueeze(0).to(device)
            outputs = model(inputs)
            
            # Save the visualization
            plt.imshow(outputs.squeeze().cpu().numpy(), cmap='gray')
            plt.savefig(f"{config['paths']['visualization_output']}/segmentation_{i}.png")
            plt.close()

if __name__ == "__main__":
    main()
