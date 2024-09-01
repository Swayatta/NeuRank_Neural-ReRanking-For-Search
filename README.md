# Deep Learning Project: Amazon ESCI Dataset

This project trains a BERT-based model on the Amazon ESCI dataset for query-document relevance prediction.

## Project Structure
root/
│
├── src/
│ ├── data/
│ ├── model/
│ ├── training/
│ └── utils/
│
├── configs/
├── scripts/
│ ├── run_training.sh
│ ├── run_tensorboard.sh
│ └── kill_tensorboard.sh
│
├── notebooks/
├── tests/
├── outputs/
│ ├── logs/
│ ├── models/
│ └── results/
│
├── main.py
├── requirements.txt
└── README.md


## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the training:
   ```
   ./scripts/run_training.sh
   ```

3. In a separate terminal, start TensorBoard:
   ```
   ./scripts/run_tensorboard.sh
   ```

4. To stop TensorBoard when you're done:
   ```
   ./scripts/kill_tensorboard.sh
   ```

## Components

- `src/data/data_loader.py`: Handles dataset loading and preprocessing
- `src/model/model.py`: Defines the model architecture
- `src/training/trainer.py`: Contains the training loop and evaluation logic
- `src/utils/utils.py`: Utility functions and configurations
- `main.py`: Entry point of the project
- `scripts/run_training.sh`: Script to start the training process
- `scripts/run_tensorboard.sh`: Script to start TensorBoard for visualizing training logs
- `scripts/kill_tensorboard.sh`: Script to kill all TensorBoard processes

## Results

The trained model will be saved in the `outputs/models` directory. Training logs can be found in the `outputs/logs` directory, and you can visualize them using TensorBoard.

These files form the core of your project. Remember to make the script files executable using `chmod +x scripts/*.sh` before running them.
