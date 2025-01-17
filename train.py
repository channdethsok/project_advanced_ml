import os
import logging
import argparse
import torch
import pandas as pd
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from src.data import prepare_dataloaders
from src.model import LyricsGenerator
from src.utils import initialize_lyrics_tokenizer, initialize_midi_tokenizer,train


def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(save_dir, "training.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Lyrics Generator Model")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to the dataset directory")
    parser.add_argument("--save_dir", type=str, default="logging",
                        help="logging")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training and validation")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate for the optimizer")
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                        help="number of warm up step")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on (e.g., 'cuda' or 'cpu')")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Log
    setup_logging(args.save_dir)
    logging.info("Training started.")

    # Tokenizers
    lyrics_tokenizer = initialize_lyrics_tokenizer()
    midi_tokenizer = initialize_midi_tokenizer()

    # Data
    try:
        df = pd.read_csv('data/lyrics_midi_data.csv')
        train_dataloader, val_dataloader = prepare_dataloaders(
            df=df,
            lyrics_tokenizer=lyrics_tokenizer,
            midi_tokenizer=midi_tokenizer,
            max_length=args.max_length,
            root_dir=args.data_dir,
            batch_size=args.batch_size,
            # subset_size=100,
        )
        logging.info("DataLoaders prepared successfully.")
    except Exception as e:
        logging.error(f"Error preparing DataLoaders: {e}")
        raise

    # Model
    try:
        model = LyricsGenerator(
            lyrics_tokenizer=lyrics_tokenizer,
            d_model=768,
            max_lyrics_length=args.max_length,
            max_midi_length=args.max_length
        )
        logging.info("Model initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing model: {e}")
        raise

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dataloader) * args.epochs
    scheduler =get_linear_schedule_with_warmup(optimizer, num_training_steps=total_steps, num_warmup_steps=args.num_warmup_steps)

    # Training
    try:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        train(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            device=device,
            lyrics_tokenizer=lyrics_tokenizer,
            midi_tokenizer=midi_tokenizer,
        )
        logging.info("Training completed successfully.")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
