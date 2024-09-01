from transformers import Trainer
from sklearn.metrics import mean_squared_error
import os

def train_and_evaluate(model, train_dataset, test_dataset, training_args, device):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating the model...")
    final_results = trainer.evaluate()
    print(f"Evaluation results: {final_results}")

    print("Saving the model...")
    model_save_path = os.path.join("outputs", "models", "final_model")
    trainer.save_model(model_save_path)
    print(f"Model saved successfully at {model_save_path}")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    return {"mse": mse}