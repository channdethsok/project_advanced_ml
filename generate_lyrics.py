import os
import argparse
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from src.model import LyricsGenerator
from src.utils import initialize_lyrics_tokenizer, initialize_midi_tokenizer, generate_lyrics, load_checkpoint
from src.data import prepare_dataloaders


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate lyrics conditioned on MIDI input.")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to the lyrics_midi_dataset directory")
    parser.add_argument("--checkpoint", type=str, default="model_checkpoint/best_checkpoint.pth", help="Path to the trained model checkpoint")
    parser.add_argument("--midi_index", type=int, default=0, help="Index of the MIDI file in the dataset")
    parser.add_argument("--max_midi_length", type=int, default=512, help="Maximum length of MIDI token sequence")
    parser.add_argument("--max_lyrics_length", type=int, default=512, help="Maximum length of lyrics sequence")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Validate dataset
    dataset_path = os.path.join(args.data_dir, "lyrics_midi_data.csv")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    df = pd.read_csv(dataset_path)

    # Validate checkpoint
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint}")

    # Initialize tokenizers
    lyrics_tokenizer = initialize_lyrics_tokenizer()
    midi_tokenizer = initialize_midi_tokenizer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model and load checkpoint
    model = LyricsGenerator(
        lyrics_tokenizer=lyrics_tokenizer,
        d_model=768,
        max_lyrics_length=args.max_lyrics_length,
        max_midi_length=args.max_midi_length,
    )

    model = load_checkpoint(model=model, path=args.checkpoint, inference=True, device=device)
    model.eval()

    print("\nModel loaded successfully. Ready for MIDI input.")

    while True:
        midi_index = input("\nEnter MIDI index (or type 'exit' to quit): ")
        if midi_index.lower() == 'exit':
            print("Exiting----")
            break

        try:
            midi_index = int(midi_index)
            if midi_index >= len(df):
                raise IndexError(f"MIDI index {midi_index} is out of range. Dataset has {len(df)} entries.")
            midi_path = os.path.join(args.data_dir, df["midi_path"][midi_index])
        except (ValueError, IndexError) as e:
            print(f"Invalid input: {e}")
            continue

        input_text = input("Enter optional input text for conditioning (or press Enter to skip): ")
        
        # Generate lyrics
        try:
            generated_lyrics = generate_lyrics(
                model=model,
                midi_path=midi_path,
                lyrics_tokenizer=lyrics_tokenizer,
                midi_tokenizer=midi_tokenizer,
                max_midi_length=args.max_midi_length,
                max_lyrics_length=args.max_lyrics_length,
                num_beams=args.num_beams,
                input_text=input_text if input_text.strip() else None
            )
            print("\nGenerated Lyrics:")
            print(generated_lyrics)
        except Exception as e:
            print(f"Error generating lyrics: {e}")


if __name__ == "__main__":
    main()
