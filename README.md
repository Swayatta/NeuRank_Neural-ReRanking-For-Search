# NeuRank: Neural Re-ranking for Search

This project focuses on developing an advanced re-ranking model using the Amazon ESCI dataset. By leveraging deep learning techniques, we aim to significantly improve e-commerce search result relevance and user experience.

![Amazon ESCI Dataset](https://m.media-amazon.com/images/G/01/mobile-apps/dex/alexa/alexa-skills-kit/tutorials/quiz-game/header._TTH_.png)

This project trains a BERT-based model on the Amazon ESCI dataset for query-document relevance prediction. By leveraging advanced natural language processing techniques, we aim to improve e-commerce search accuracy and efficiency.

Key features:
- 🔍 Query-document relevance prediction
- 🤖 BERT-based model architecture
- 📊 Amazon ESCI dataset utilization

Join us in exploring the intersection of deep learning and e-commerce search optimization!

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



## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch training:**
   ```bash
   ./scripts/run_training.sh
   ```

3. **Visualize with TensorBoard:**
   ```bash
   ./scripts/run_tensorboard.sh
   ```

4. **Stop TensorBoard:**
   ```bash
   ./scripts/kill_tensorboard.sh
   ```

## 🧩 Key Components

- 🔍 `src/data/data_loader.py`: Dataset magic happens here
- 🧠 `src/model/model.py`: BERT-based architecture defined
- 🏋️ `src/training/trainer.py`: Training loop and evaluation logic
- 🛠️ `src/utils/utils.py`: Handy utility functions
- 🚪 `main.py`: Project entry point

## 📊 Results & Outputs

- 💾 Trained models: `outputs/models`
- 📈 Training logs: `outputs/logs`
- 🖥️ Visualize: TensorBoard

## 💡 Pro Tips

- Make scripts executable: `chmod +x scripts/*.sh`
- Experiment with hyperparameters in `configs/config.yaml`
- Check out `notebooks/` for data exploration

## 🤝 Contributing

We welcome contributions! Please check out our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Happy Training! 🎉</strong>
</p>
