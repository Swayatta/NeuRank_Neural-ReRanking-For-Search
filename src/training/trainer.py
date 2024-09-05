import os
import yaml
import mlflow
from sklearn.metrics import mean_squared_error
from transformers import Trainer
from src.utils.utils import get_training_args
from src.utils.callbacks import LossPrintCallback, MLflowCallback
from src.loss.loss_functions import mse_loss, bce_loss
class CustomTrainer(Trainer):
    def __init__(self, model, args, loss_function, train_dataset=None, eval_dataset=None, compute_metrics=None, callbacks=None):
        super().__init__(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics, callbacks=callbacks)
        self.loss_function = loss_function

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_function(logits, labels)
        return (loss, outputs) if return_outputs else loss

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_and_evaluate(model, train_dataset, test_dataset, device):
    config = load_config('configs/config.yaml')
    mlflow.set_experiment(config['experiment_name'])

    # Get training arguments from config
    training_args = get_training_args(config['training_args'])

    # Log parameters
    mlflow.log_params({
        "model_type": config['model_type'],
        **config['training_args']
    })
    if config['loss_function'] == 'mse':
        loss_function = mse_loss
    elif config['loss_function'] == 'bce':
        loss_function = bce_loss
    else:
        raise ValueError(f"Invalid loss function: {config['loss_function']}")
    trainer = CustomTrainer(
        model=model,
        loss_function=loss_function,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[LossPrintCallback(), MLflowCallback()],
    )

    # Debug: Print the first batch
    for batch in train_dataset:
        print("Batch keys:", batch.keys())
        for key, value in batch.items():
            print(f"{key} shape:", value.shape)
        break

    print("Training the model...")
    trainer.train()

    print("Evaluating the model...")
    final_results = trainer.evaluate()
    print(f"Evaluation results: {final_results}")

    # Log metrics
    mlflow.log_metrics(final_results)

    print("Saving the model...")
    model_save_path = os.path.join("outputs", "models", "final_model")
    trainer.save_model(model_save_path)
    print(f"Model saved successfully at {model_save_path}")
    mlflow.log_artifacts(model_save_path, artifact_path="model")
 

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    return {"mse": mse}