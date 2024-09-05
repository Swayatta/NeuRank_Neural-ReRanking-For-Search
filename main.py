import torch
from src.data.data_loader import load_and_prepare_data
from src.model.model import get_model
from src.training.trainer import train_and_evaluate
# Remove this import as we no longer need it here
# from src.utils.utils import get_training_args

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, test_dataset = load_and_prepare_data()
    model = get_model(device)
    
    # Remove this line as we no longer need to get training args here
    # training_args = get_training_args()

    # Update the function call to remove training_args
    train_and_evaluate(model, train_dataset, test_dataset, device)

if __name__ == "__main__":
    main()