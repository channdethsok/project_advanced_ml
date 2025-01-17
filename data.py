import random
from copy import deepcopy
from symusic import Note, Score
import torch
from torch.utils.data import Dataset, DataLoader
import os

def randomize_midi_pitch(midi_score, prob=0.2, max_change=4):
    new_score = deepcopy(midi_score)
    for track in new_score.tracks:
        for note in track.notes:
            if random.random() < prob:
                change = random.randint(-max_change, max_change)
                note.pitch = max(0, min(note.pitch + change, 127))
    return new_score


class LyricsMidiDataset(Dataset):
    def __init__(self, dataframe, lyrics_tokenizer, midi_tokenizer, max_length, root_dir=None, augment=True):
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
        # # Debugging checks
        # print("Lyrics:", lyrics)
        # print("Lyrics Tokens:", lyrics_tokens['input_ids'])
        # print("MIDI Path:", midi_path)
        # print("MIDI Tokens before padding:", midi_tokens)

        return {
            'lyrics_ids': lyrics_tokens['input_ids'].squeeze(0),
            'lyrics_attention_mask': lyrics_tokens['attention_mask'].squeeze(0),
            'midi_tokens': midi_tokens
        }

    def _pad_or_truncate(self, tokens):
        # Truncate sequences longer than `max_length`
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        # Pad sequences shorter than `max_length`
        elif len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)