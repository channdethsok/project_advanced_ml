from transformers import AutoModel, GPT2LMHeadModel
import torch.nn as nn
import torch
from symusic import Score

class LyricsGenerator(nn.Module):
    def __init__(self, lyrics_vocab_size, d_model, max_lyrics_length, max_midi_length):
        super(LyricsGenerator, self).__init__()

        # MIDI Encoder
        self.midi_encoder = AutoModel.from_pretrained("ruru2701/musicbert-v1.1")
        self.midi_projection = nn.Linear(self.midi_encoder.config.hidden_size, d_model)
        self.midi_positional_embedding = nn.Embedding(max_midi_length, d_model)

        # GPT-2 for lyrics
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=50257)
        self.gpt2.resize_token_embeddings(lyrics_vocab_size)
        self.lyrics_positional_embedding = nn.Embedding(max_lyrics_length, d_model)

        # Cross-Attention Layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=8)

    def forward(self, lyrics_ids, lyrics_attention_mask, midi_tokens):
        # MIDI Encoding
        midi_attention_mask = (midi_tokens != 0).int()
        midi_outputs = self.midi_encoder(input_ids=midi_tokens, attention_mask=midi_attention_mask)
        midi_embeds = self.midi_projection(midi_outputs.last_hidden_state)
        midi_positions = torch.arange(midi_tokens.size(1), device=midi_tokens.device).unsqueeze(0)
        midi_embeds += self.midi_positional_embedding(midi_positions)

        # Lyrics Encoding
        lyrics_positions = torch.arange(lyrics_ids.size(1), device=lyrics_ids.device).unsqueeze(0)
        lyrics_embeds = self.gpt2.transformer.wte(lyrics_ids) + self.lyrics_positional_embedding(lyrics_positions)

        # Masking Lyrics Embeddings Before Cross-Attention
        lyrics_embeds_masked = lyrics_embeds * lyrics_attention_mask.unsqueeze(-1)

        # Cross-Attention
        midi_embeds_t = midi_embeds.transpose(0, 1)
        lyrics_embeds_t = lyrics_embeds_masked.transpose(0, 1)
        cross_attn_output, _ = self.cross_attention(
            query=lyrics_embeds_t,
            key=midi_embeds_t,
            value=midi_embeds_t,
        )
        combined_embeds = lyrics_embeds + cross_attn_output.transpose(0, 1)

        # Concatenate and Pass through GPT-2
        combined_attention_mask = torch.cat(
            [torch.ones((lyrics_ids.size(0), midi_tokens.size(1)), device=lyrics_ids.device), lyrics_attention_mask],
            dim=1
        )
        combined_embeds = torch.cat((midi_embeds, combined_embeds), dim=1)
        outputs = self.gpt2(inputs_embeds=combined_embeds, attention_mask=combined_attention_mask)
        return outputs.logits[:, midi_tokens.size(1):, :]


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
    Generates lyrics conditioned on MIDI input and optional input text using the trained model.
    
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

    # Tokenize MIDI
    try:
        midi_score = Score(midi_path)  # Load MIDI file into Score object
        midi_tokens = midi_tokenizer.encode(midi_score)[0].ids  # Tokenize MIDI
    except Exception as e:
        raise ValueError(f"Error processing MIDI file: {e}")
    
    # Pad or truncate MIDI tokens to max_midi_length
    midi_tokens = midi_tokens[:max_midi_length]
    padding_length = max_midi_length - len(midi_tokens)
    midi_tokens = midi_tokens + [0] * padding_length  # Pad with 0s
    midi_tokens = torch.tensor(midi_tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Initialize input for lyrics generation
    if input_text:
        # Tokenize
        input_ids = lyrics_tokenizer.encode(input_text, return_tensors="pt").to(device)
    else:
        # Default to starting token if no input text
        input_ids = torch.tensor(lyrics_tokenizer.encode("<|endoftext|>")).unsqueeze(0).to(device)
    
    attention_mask = torch.ones_like(input_ids).to(device)

    # Generate with beam search
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