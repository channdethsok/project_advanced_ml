import random
from copy import deepcopy
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from symusic import Score


def randomize_midi_pitch(midi_score, prob=0.2, max_change=4):
    """
    Randomly changes the pitch of notes in a MIDI score.

    Args:
        midi_score (Score): A Score object representing the MIDI file.
        prob (float): Probability of changing a note's pitch.
        max_change (int): Maximum semitone change (+/-).

    Returns:
        Score: The modified Score object.
    """
    new_score = deepcopy(midi_score)
    for track in new_score.tracks:
        for note in track.notes:
            if random.random() < prob:
                change = random.randint(-max_change, max_change)
                note.pitch = max(0, min(note.pitch + change, 127))
    return new_score


class LyricsMidiDataset(Dataset):
    """
    lyrics and MIDI data

    Args:
        dataframe (pd.DataFrame): DataFrame with 'lyrics' and 'midi_path' columns.
        lyrics_tokenizer (Tokenizer): Tokenizer for lyrics.
        midi_tokenizer (Tokenizer): Tokenizer for MIDI files.
        max_length (int): Maximum sequence length for tokenization.
        root_dir (str, optional): Root directory for MIDI files.
        augment (bool, optional): Whether to augment MIDI data.
    """
    def __init__(self, dataframe, lyrics_tokenizer, midi_tokenizer,
                 max_length, root_dir=None, augment=True):
        self.dataframe = dataframe
        self.lyrics_tokenizer = lyrics_tokenizer
        self.midi_tokenizer = midi_tokenizer
        self.max_length = max_length
        self.augment = augment
        self.root_dir = root_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        lyrics = self.dataframe.iloc[idx]['lyrics']
        midi_path = self.dataframe.iloc[idx]['midi_path']

        if self.root_dir:
            midi_path = os.path.join(self.root_dir, midi_path)

        midi_path = os.path.normpath(midi_path)
        if not os.path.isfile(midi_path):
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")

        lyrics_tokens = self.lyrics_tokenizer(
            lyrics + self.lyrics_tokenizer.eos_token,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        midi_score = Score(midi_path)
        if self.augment:
            midi_score = randomize_midi_pitch(midi_score)

        midi_tokens = self.midi_tokenizer.encode(midi_score)[0].ids
        midi_tokens = self._pad_or_truncate(midi_tokens)

        return {
            'lyrics_ids': lyrics_tokens['input_ids'].squeeze(0),
            'lyrics_attention_mask': lyrics_tokens['attention_mask'].squeeze(0),
            'midi_tokens': midi_tokens
        }

    def _pad_or_truncate(self, tokens):
        """
        Pads or truncates a sequence of tokens to a fixed length.

        Args:
            tokens (list[int]): The token sequence.

        Returns:
            torch.Tensor: A tensor of the padded or truncated tokens.
        """
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        elif len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)


def prepare_dataloaders(df, lyrics_tokenizer, midi_tokenizer, max_length,
                        root_dir, batch_size=4, subset_size=None):
    """
    Prepares DataLoaders for training and validation.

    Args:
        df (pd.DataFrame): The dataset DataFrame.
        lyrics_tokenizer (Tokenizer): Tokenizer for lyrics.
        midi_tokenizer (Tokenizer): Tokenizer for MIDI files.
        max_length (int): Maximum sequence length for tokenization.
        root_dir (str): Root directory for MIDI files.
        batch_size (int, optional): Batch size for DataLoaders. Defaults to 4.
        subset_size (int, optional): Size of the subset for quick testing.

    Returns:
        tuple: Training and validation DataLoaders.
    """
    dataset = LyricsMidiDataset(
        dataframe=df,
        lyrics_tokenizer=lyrics_tokenizer,
        midi_tokenizer=midi_tokenizer,
        max_length=max_length,
        root_dir=root_dir,
        augment=True
    )

    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(
        dataset, [train_size, len(dataset) - train_size]
    )

    if subset_size:
        train_dataset = Subset(train_dataset, range(subset_size))
        val_dataset = Subset(val_dataset, range(subset_size))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader
