import os
import argparse
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from src.model import LyricsGenerator
from src.utils import initialize_lyrics_tokenizer, initialize_midi_tokenizer, generate_lyrics, load_checkpoint


def parse_arguments():
    """
    Parses command-line arguments for the lyrics generation script.
    """
    parser = argparse.ArgumentParser(description="Generate lyrics conditioned on MIDI input.")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to the lyrics_midi_dataset directory")
    parser.add_argument("--checkpoint", type=str, default="model_checkpoint/final_checkpoint.pth", help="Path to the trained model checkpoint")
    parser.add_argument("--midi_index", type=int, default=0, help="Index of the MIDI file in the dataset")
    parser.add_argument("--max_midi_length", type=int, default=512, help="Maximum length of MIDI token sequence")
    parser.add_argument("--max_lyrics_length", type=int, default=512, help="Maximum length of lyrics sequence")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
    parser.add_argument("--input_text", type=str, default=None, help="Optional input text to condition lyrics generation")
    return parser.parse_args()

def main():
    args = parse_arguments()

    dataset_path = os.path.join(args.data_dir, "lyrics_midi_data.csv")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    df = pd.read_csv(dataset_path)

    if args.midi_index >= len(df):
        raise IndexError(f"MIDI index {args.midi_index} is out of range. Dataset has {len(df)} entries.")

    midi_path = os.path.join(args.data_dir, df["midi_path"][args.midi_index])

    # tokenizers
    lyrics_tokenizer = initialize_lyrics_tokenizer()
    midi_tokenizer = initialize_midi_tokenizer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = LyricsGenerator(
        lyrics_tokenizer=lyrics_tokenizer,
        d_model=768,
        max_lyrics_length=args.max_lyrics_length,
        max_midi_length=args.max_midi_length,
    )
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint}")
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    model, optimizer, scheduler, start_epoch = load_checkpoint(model, optimizer, scheduler, args.checkpoint, device)
    model.eval()

    # Generate lyrics
    generated_lyrics = generate_lyrics(
        model=model,
        midi_path=midi_path,
        lyrics_tokenizer=lyrics_tokenizer,
        midi_tokenizer=midi_tokenizer,
        max_midi_length=args.max_midi_length,
        max_lyrics_length=args.max_lyrics_length,
        num_beams=args.num_beams,
        input_text=args.input_text
    )

    print("\nGenerated Lyrics:")
    print(generated_lyrics)


if __name__ == "__main__":
    main()
