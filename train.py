import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from datetime import datetime


print(torch.cuda.is_available())
print(torch.version.cuda)
# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
try:
    dataset = load_dataset("nixiesearch/amazon-esci", streaming=True)
    print("Dataset loaded in streaming mode successfully.")
except Exception as e:
    print(f"Error loading dataset in streaming mode: {e}")
    raise

# Take samples from train and test splits
train_samples = list(dataset["train"].take(5))
test_samples = list(dataset["test"].take(5))

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Function to tokenize and prepare samples for training
def prepare_samples(examples):
    samples = []
    
    # Create positive samples
    for query, doc in zip(examples["query"], examples["doc"]):
        samples.append({"query": query, "text": doc, "label": 1.0})
    
    # Create negative samples
    for query, negs, negscores in zip(examples["query"], examples["neg"], examples["negscore"]):
        for neg, negscore in zip(negs, negscores):
            samples.append({"query": query, "text": neg, "label": negscore})
    
    # Tokenize the batch
    tokenized = tokenizer(
        [sample["query"] for sample in samples],
        [sample["text"] for sample in samples],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Add labels
    tokenized["labels"] = torch.tensor([sample["label"] for sample in samples])
    
    return tokenized

# Convert samples to Dataset objects and tokenize
train_dataset = Dataset.from_list(train_samples)
test_dataset = Dataset.from_list(test_samples)

tokenized_train_dataset = train_dataset.map(
    prepare_samples,
    batched=True,
    batch_size=32,
    remove_columns=train_dataset.column_names
)

tokenized_test_dataset = test_dataset.map(
    prepare_samples,
    batched=True,
    batch_size=32,
    remove_columns=test_dataset.column_names
)

# Initialize the model and move it to the device
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
model.to(device)

print("Training dataset size:", len(tokenized_train_dataset))
print("Test dataset size:", len(tokenized_test_dataset))

# Define custom metrics computation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    return {"mse": mse}

# Define training arguments
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir=f"./logs/{timestamp}",
    evaluation_strategy="epoch",
    save_strategy="no",
    load_best_model_at_end=False,
    logging_strategy="epoch",  # Log at the end of each epoch
    report_to="tensorboard",  # Enable tensorboard logging
    no_cuda=False,  # Enable GPU usage
    learning_rate=0.0001,
    lr_scheduler_type="constant",
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Start training
print("Starting training...")
trainer.train()

# Evaluate the model
print("Evaluating the model...")
# After training, log the final evaluation results
final_results = trainer.evaluate()

print(f"Evaluation results: {final_results}")

# Save the model
print("Saving the model...")
trainer.save_model("./final_model")
print("Model saved successfully.")