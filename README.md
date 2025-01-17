# Lyrics Generator Conditioned on MIDI Music

This repository contains code for training a lyrics generation model that is conditioned on MIDI music. The model uses a combination of MusicBERT (for encoding MIDI) and GPT-2 (for generating lyrics), connected through a cross-attention mechanism.

## Repository Structure

The repository is organized as follows:

*   **`train.py`:** Contains the main training loop, including validation and checkpoint saving/loading.
*   **`model.py`:** Defines the `LyricsGenerator` model architecture, including the MusicBERT encoder, GPT-2 decoder, and cross-attention layer.
*   **`data.py`:** Contains data loading and preprocessing utilities, including the `MIDILyricsDataset` class for creating PyTorch datasets.
*   **`demo.ipynb`:** A Jupyter Notebook demonstrating how to train and use the trained model to generate lyrics from a MIDI file.
*   **`data/`:**  A folder that will contain the lyrics and MIDI datasets.
    *   **`preprocess.ipynb`:** A notebook showing how the raw data was preprocessed to create the dataset.
*   **`tokenizer/`:**
    *   **`tokenizer.json`:** A JSON file containing the trained tokenizer used for MIDI data.
