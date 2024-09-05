from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import torch

def load_and_prepare_data():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    try:
        # dataset = load_dataset("nixiesearch/amazon-esci", streaming=True)
        train_dataset = load_dataset("nixiesearch/amazon-esci", streaming=True, split="train")
        test_dataset = load_dataset("nixiesearch/amazon-esci", streaming=True, split="test")
        print("Dataset loaded in streaming mode successfully.")
    except Exception as e:
        print(f"Error loading dataset in streaming mode: {e}")
        raise

    # train_samples = list(dataset["train"].take(5))
    # test_samples = list(dataset["test"].take(5))

    # train_dataset = Dataset.from_list(train_samples)
    # test_dataset = Dataset.from_list(test_samples)

    tokenized_train_dataset = train_dataset.map(
        lambda examples: prepare_samples(examples, tokenizer),
        batched=True,
        batch_size=32,
        remove_columns=train_dataset.column_names
    )

    tokenized_test_dataset = test_dataset.map(
        lambda examples: prepare_samples(examples, tokenizer),
        batched=True,
        batch_size=32,
        remove_columns=test_dataset.column_names
    )

    return tokenized_train_dataset, tokenized_test_dataset

def prepare_samples(examples, tokenizer):
    samples = []
    
    for query, doc, negs, negscores in zip(examples["query"], examples["doc"], examples["neg"], examples["negscore"]):
        samples.append({"query": query, "text": doc, "label": 1.0})
        for neg, negscore in zip(negs, negscores):
            samples.append({"query": query, "text": neg, "label": negscore})
    
    tokenized = tokenizer(
        [sample["query"] for sample in samples],
        [sample["text"] for sample in samples],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    tokenized["labels"] = torch.tensor([sample["label"] for sample in samples])
    
    return tokenized