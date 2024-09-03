import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.vnet import VNet
from dataset import CTAbdomenDataset  # Assuming you have a custom dataset class
import yaml

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Data loaders
    train_dataset = CTAbdomenDataset(config['paths']['train_data'], transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}, Loss: {loss.item()}")
    
    # Save model
    torch.save(model.state_dict(), config['paths']['model_save_path'])

if __name__ == "__main__":
    main()
