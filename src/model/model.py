from transformers import AutoModelForSequenceClassification

def get_model(device):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
    return model.to(device)