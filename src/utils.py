from transformers import GPT2Tokenizer
from miditok import TSD, TokenizerConfig
from pathlib import Path
import os
from tqdm import tqdm
from symusic import Score
import torch
from torch import nn
from torch.amp import autocast, GradScaler
import logging

log_dir = "logging"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "training.log"),
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

def initialize_lyrics_tokenizer():
    lyrics_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    lyrics_tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    return lyrics_tokenizer

def initialize_midi_tokenizer(tokenizer_file="tokenizer/tokenizer.json"):
    config = TokenizerConfig(
        use_velocities=False,
        num_velocities=1,
        use_chords=False,
        use_rests=False,
        use_tempos=False,
        use_time_signatures=False,
    )
    midi_tokenizer = TSD(config)
    return midi_tokenizer.from_pretrained(Path(tokenizer_file))
    # return midi_tokenizer

def train(model, train_dataloader, val_dataloader, optimizer, scheduler,
          epochs, device, lyrics_tokenizer,midi_tokenizer, save_every=None):
    """
    Trains the LyricsGenerator model.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        optimizer (Optimizer): Optimizer for training.
        scheduler (Scheduler): Learning rate scheduler.
        epochs (int): Number of epochs to train.
        device (torch.device): Device to use for training (e.g., 'cuda').
        lyrics_tokenizer (Tokenizer): Tokenizer for lyrics.
        save_every (int, optional): Save model checkpoints every `save_every` epochs. If None, only saves the best and final checkpoints.
    """
    model.to(device)
    scaler = GradScaler()
    loss_fct = nn.CrossEntropyLoss(ignore_index=lyrics_tokenizer.pad_token_id)
    save_dir = "model_checkpoint"
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    midi_path = "./data/lmd-full_and_reddit_MIDI_dataset/sentenceWord_level_6_MIDI/0a1351c0d893782fa7a6b16e43e391b2.mid"
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        loop = tqdm(train_dataloader, leave=True)
        for i, batch in enumerate(loop):
            optimizer.zero_grad()

            lyrics_ids = batch['lyrics_ids'].to(device)
            lyrics_attention_mask = batch['lyrics_attention_mask'].to(device)
            midi_tokens = batch['midi_tokens'].to(device)

            with autocast(device_type=device.type):
                logits = model(lyrics_ids, lyrics_attention_mask, midi_tokens)
                loss = loss_fct(logits.transpose(1, 2), lyrics_ids)
                train_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item())

        val_loss = validate(model, val_dataloader, loss_fct, device)
        logging.info(
            f"Epoch {epoch + 1}/{epochs}: "
            f"Training Loss: {train_loss / len(train_dataloader):.4f}, Validation Loss: {val_loss:.4f}"
        )
        # print(f"Epoch {epoch + 1}/{epochs}, "
        #       f"Training Loss: {train_loss / len(train_dataloader):.4f}, "
        #       f"Validation Loss: {val_loss:.4f}")

        print(f"Generating lyrics after epoch {epoch + 1}...")
        generated_lyrics = generate_lyrics(
                model=model,
                midi_path=midi_path,
                lyrics_tokenizer=lyrics_tokenizer,
                midi_tokenizer=midi_tokenizer,
                max_midi_length=512,
                max_lyrics_length=512
            )
        logging.info(f"Generated Lyrics (Epoch {epoch + 1}): {generated_lyrics}")
        # Save the best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, save_dir, filename="best_checkpoint.pth")
            print(f"Best checkpoint saved for epoch {epoch + 1} with val_loss: {val_loss:.4f}!")

        # Save intermediate checkpoints if save_every is specified
        if save_every and (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, save_dir)
            print(f"Checkpoint saved for epoch {epoch + 1}!")

    # Save the final
    save_checkpoint(model, optimizer, scheduler, epochs, save_dir, filename="final_checkpoint.pth")
    print("Final checkpoint saved!")

def validate(model, dataloader, loss_fct, device):
    """
    Validates the model on the validation dataset.

    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): DataLoader for validation data.
        loss_fct (Loss): Loss function.
        device (torch.device): Device to use for validation.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            lyrics_ids = batch['lyrics_ids'].to(device)
            lyrics_attention_mask = batch['lyrics_attention_mask'].to(device)
            midi_tokens = batch['midi_tokens'].to(device)

            with autocast(device_type=device.type):
                logits = model(lyrics_ids, lyrics_attention_mask, midi_tokens)
                loss = loss_fct(logits.transpose(1, 2), lyrics_ids)
                val_loss += loss.item()

    return val_loss / len(dataloader)


def save_checkpoint(model, optimizer, scheduler, epoch, save_dir, filename=None):
    """
    Saves the model checkpoint.

    Args:
        model (nn.Module): Model to save.
        optimizer (Optimizer): Optimizer state to save.
        scheduler (Scheduler): Scheduler state to save.
        epoch (int): Current epoch number.
        save_dir (str): Directory to save the checkpoint.
        filename (str, optional): Filename for the checkpoint. Defaults to "checkpoint_epoch_{epoch}.pt".
    """
    os.makedirs(save_dir, exist_ok=True)
    if filename is None:
        filename = f"checkpoint_epoch_{epoch + 1}.pt"
    checkpoint_path = os.path.join(save_dir, filename)
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")

def load_checkpoint(model, optimizer=None, scheduler=None, path=None, device=None, inference=True):
    if path is None:
        raise ValueError("Checkpoint path must be specified.")
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    if inference:
        print("Checkpoint loaded for inference.")
        model.eval()
        return model
    else:
        if optimizer is None or scheduler is None:
            raise ValueError("Optimizer and scheduler must be provided for training mode.")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded. Resuming from epoch {epoch}.")
        return model, optimizer, scheduler, epoch


def generate_lyrics(
    model,
    midi_path,
    lyrics_tokenizer,
    midi_tokenizer,
    max_midi_length,
    max_lyrics_length,
    num_beams=5,
    input_text=None
):
    """
    Generates lyrics conditioned on MIDI input and optional input text using the trained model
    Args:
        model: Trained LyricsGenerator model.
        midi_path: Path to the MIDI file.
        lyrics_tokenizer: Tokenizer for lyrics (e.g., GPT-2 tokenizer).
        midi_tokenizer: Tokenizer for MIDI (e.g., miditok TSD tokenizer).
        max_midi_length: Maximum length of MIDI token sequence.
        max_lyrics_length: Maximum length of lyrics sequence.
        num_beams: Number of beams for beam search.
        input_text: Optional input text to condition lyrics generation.
    Returns:
        Generated lyrics as a string.
    """
    device = next(model.parameters()).device
    model.eval()

    # Check if MIDI file exists
    midi_file = Path(midi_path)
    if not midi_file.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    # Tokenize MIDI
    try:
        midi_score = Score(midi_path) 
        midi_tokens = midi_tokenizer.encode(midi_score)[0].ids
    except Exception as e:
        raise ValueError(f"Error processing MIDI file: {e}")

    # Pad or truncate MIDI tokens to max_midi_length
    midi_tokens = midi_tokens[:max_midi_length]
    padding_length = max_midi_length - len(midi_tokens)
    midi_tokens = midi_tokens + [midi_tokenizer.pad_token_id] * padding_length
    midi_tokens = torch.tensor(midi_tokens, dtype=torch.long).unsqueeze(0).to(device)

    if input_text:
        input_ids = lyrics_tokenizer.encode(input_text, return_tensors="pt").to(device)
    else:
        input_ids = torch.tensor(lyrics_tokenizer.encode("<|endoftext|>")).unsqueeze(0).to(device)
    
    attention_mask = torch.ones_like(input_ids).to(device)

    beam_output = model.gpt2.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_lyrics_length,
        num_beams=num_beams,
        early_stopping=True,
        num_return_sequences=1,
        pad_token_id=lyrics_tokenizer.pad_token_id,
        do_sample=True
    )

    # Decode
    generated_lyrics = lyrics_tokenizer.decode(beam_output[0], skip_special_tokens=True)

    return generated_lyrics