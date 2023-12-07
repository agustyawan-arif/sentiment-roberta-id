# Sentiment Analysis with Indonesian RoBERTa

This repository contains code for training a sentiment analysis model using the Indonesian RoBERTa base model. The sentiment analysis is performed on an SmSA dataset obtained from the IndoNLU datasets.

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/agustyawan-arif/sentiment-roberta-id.git
   cd sentiment-roberta-id
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The training and model configuration are specified in the `config.conf` file. Modify this file to adjust training parameters, output directories, and other settings.

## Training

Run the training script to train the sentiment analysis model:

```bash
python src/trainer.py
```

The script uses the specified configuration file (`config.conf`) for training parameters.

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [IndoNLU Datasets](https://huggingface.co/datasets/indonlp/indonlu)