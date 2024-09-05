from transformers import TrainerCallback
import mlflow

class LossPrintCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model, **kwargs):
        if state.log_history:
            loss = state.log_history[-1].get('loss')
            print(f"Loss = {loss}")
        else:
            print("No log history available")

class MLflowCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if mlflow.active_run():
            mlflow.end_run()
        mlflow.start_run()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for key, value in logs.items():
                mlflow.log_metric(key, value, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        mlflow.end_run()
