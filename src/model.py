from transformers import AutoModel, GPT2LMHeadModel
import torch.nn as nn
import torch
from symusic import Score


class LyricsGenerator(nn.Module):
    """
    A model that combines a MIDI encoder (MusicBERT) with GPT-2 for generating
    lyrics based on MIDI input.

    Args:
        lyrics_tokenizer (Tokenizer): Tokenizer for lyrics.
        d_model (int): Dimension of the embeddings.
        max_lyrics_length (int): Maximum length of the lyrics sequence.
        max_midi_length (int): Maximum length of the MIDI sequence.
    """
    def __init__(self, lyrics_tokenizer, d_model, max_lyrics_length, max_midi_length):
        super(LyricsGenerator, self).__init__()

        self.lyrics_tokenizer = lyrics_tokenizer
        self.lyrics_pad_token = lyrics_tokenizer.pad_token_id

        # MIDI Encoder
        self.midi_encoder = AutoModel.from_pretrained("ruru2701/musicbert-v1.1")
        self.midi_projection = nn.Linear(self.midi_encoder.config.hidden_size, d_model)
        self.midi_positional_embedding = nn.Embedding(max_midi_length, d_model)

        # GPT-2 for lyrics
        self.gpt2 = GPT2LMHeadModel.from_pretrained(
            'gpt2', pad_token_id=self.lyrics_pad_token
        )
        self.gpt2.resize_token_embeddings(len(lyrics_tokenizer))
        self.lyrics_positional_embedding = nn.Embedding(max_lyrics_length, d_model)

        # Cross-Attention Layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=8)

    def forward(self, lyrics_ids, lyrics_attention_mask, midi_tokens):
        """
        Args:
            lyrics_ids (torch.Tensor): Tokenized lyrics input IDs.
            lyrics_attention_mask (torch.Tensor): Attention mask for lyrics.
            midi_tokens (torch.Tensor): Tokenized MIDI sequence.

        Returns:
            torch.Tensor: Logits for the next token prediction.
        """
        # MIDI Encoding
        midi_attention_mask = (midi_tokens != 0).int()
        midi_outputs = self.midi_encoder(input_ids=midi_tokens, attention_mask=midi_attention_mask)
        midi_embeds = self.midi_projection(midi_outputs.last_hidden_state)
        midi_positions = torch.arange(midi_tokens.size(1), device=midi_tokens.device).unsqueeze(0)
        midi_embeds += self.midi_positional_embedding(midi_positions)

        # Lyrics Encoding
        lyrics_positions = torch.arange(
            lyrics_ids.size(1), device=lyrics_ids.device
        ).unsqueeze(0)
        lyrics_embeds = (
            self.gpt2.transformer.wte(lyrics_ids) +
            self.lyrics_positional_embedding(lyrics_positions)
        )

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
            [
                torch.ones(
                    (lyrics_ids.size(0), midi_tokens.size(1)),
                    device=lyrics_ids.device
                ),
                lyrics_attention_mask,
            ],
            dim=1
        )
        combined_embeds = torch.cat((midi_embeds, combined_embeds), dim=1)
        outputs = self.gpt2(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask
        )
        return outputs.logits[:, midi_tokens.size(1):, :]
