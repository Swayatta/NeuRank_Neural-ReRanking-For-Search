from transformers import TrainingArguments
from datetime import datetime
import os

def get_training_args():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", "results")
    logging_dir=os.path.join("outputs", "logs", timestamp)
    print(f"Logging directory: {logging_dir}")
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir=logging_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        logging_strategy="epoch",
        report_to="tensorboard",
        no_cuda=False,
        learning_rate=2e-5,
        lr_scheduler_type="constant",
    )