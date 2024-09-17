from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader, IterableDataset
import torch

#  Ensure the model is trained on a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# torch.set_default_tensor_type('cuda' if torch.cuda.is_available() else 'cpu')

class TripletsDataset(IterableDataset):
    def __init__(self, dataset, chunk_size=1000):
        self.dataset = dataset
        self.chunk_size = chunk_size

    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            query = sample["query"]
            positive = sample["doc"]
            negatives = sample["neg"]

            for negative in negatives:
                
                buffer.append(InputExample(texts=[query, positive, negative]))
                
                if len(buffer) >= self.chunk_size:
                    
                    yield from buffer
                    buffer = []
            
        if buffer:
            yield from buffer
            buffer = []

# Load the dataset with streaming
dataset = load_dataset("nixiesearch/amazon-esci", streaming=True)
train_dataset = dataset["train"]

# Create an IterableDataset
triplets_dataset = TripletsDataset(train_dataset, chunk_size=100)

# Initialize the SentenceTransformer model and move it to the GPU if available
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.to(device)

# Prepare the train dataloader
train_dataloader = DataLoader(triplets_dataset, batch_size=16)

# Define the triplet loss
train_loss = losses.TripletLoss(model=model)

# Train the model for one step
model.fit(
    [(train_dataloader, train_loss)],
    epochs=1,
    steps_per_epoch=10000,
    warmup_steps=100,
    show_progress_bar=True
)

print("Training completed for one step.")

# Optional: Save the model
model.save('output/bi-encoder-model')