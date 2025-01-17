# Lyrics Generator Conditioned on MIDI Music

This repository contains code for training a lyrics generation model that is conditioned on MIDI music. The model uses a combination of MusicBERT (for encoding MIDI) and GPT-2 (for generating lyrics), connected through a cross-attention mechanism.

## Repository Structure

The repository is organized as follows:

```
project_advanced_ml/
├── data/
│   ├── lmd-full_and_reddit_MIDI_dataset/# contain the lyrics and MIDI datasets.
│   ├── lyrics_midi_data.csv             # Dataframe lyrics and MIDI files
│   ├── preprocess.ipynb                 # A notebook showing how the raw data was preprocessed to create the dataset
├── notebooks/
│   ├── demo.ipynb                       
├── src/
│   ├── __init__.py                      # Package initialization
│   ├── data.py                          # Contains data loading and preprocessing utilities, including the `LyricsMidiDataset` class for creating PyTorch datasets
│   ├── model.py                         # Defines the `LyricsGenerator` model architecture, including the MusicBERT encoder, GPT-2 decoder, and cross-attention layer
│   ├── utils.py                         # Tokenizers, checkpoint saving,Training and validation functions
├── tokenizer/                           # MIDI tokenizer configuration
│   ├── tokenizer.json                   # A JSON file containing the trained tokenizer used for MIDI data.
├── .gitignore                    
├── generate_lyrics.py                   # For generating lyrics based on MIDI
├── requirements.txt                    
├── train_model.py                       # To train
├── data_exploration.ipynb               # Ingest Midi and lyrics (check Token)
```
