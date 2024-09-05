from transformers import TrainingArguments
from datetime import datetime
import os

def get_training_args(config_args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", "results")
    logging_dir = os.path.join("outputs", "logs", timestamp)
    print(f"Logging directory: {logging_dir}")
    
    return TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        report_to="none",
        **config_args
    )