import os
from tqdm import tqdm
import torch
from torch import nn
from torch.amp import autocast, GradScaler

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, device, lyrics_tokenizer, save_every=1):
    model.to(device)
    scaler = GradScaler()
    loss_fct = nn.CrossEntropyLoss(ignore_index=lyrics_tokenizer.pad_token_id)
    save_dir = "model_checkpoint"
    os.makedirs(save_dir, exist_ok=True)
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

            # # Debugging 
            # if i % 300 == 0:
            #     print(f"Epoch {epoch + 1}, Batch {i}/{len(train_dataloader)}")
            #     print(f"Loss: {loss.item():.4f}")

            #     # Decode input lyrics and predictions
            #     decoded_input = lyrics_tokenizer.decode(lyrics_ids[0].tolist(), skip_special_tokens=True)
            #     predicted_tokens = logits.argmax(dim=-1)[0]
            #     decoded_prediction = lyrics_tokenizer.decode(predicted_tokens.tolist(), skip_special_tokens=True)

            #     # print(f"Input Lyrics: {decoded_input}")
            #     print("_______________________________")
            #     print(f"Predicted Lyrics: {decoded_prediction}")
            #     # print(f"MIDI Tokens: {midi_tokens[0].cpu().numpy()}")
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

        val_loss = validate(model, val_dataloader, loss_fct, device)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss / len(train_dataloader):.4f}, Validation Loss: {val_loss:.4f}")

        # Save Checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, save_dir)
            print(f"Checkpoint saved for epoch {epoch + 1}!")
    if not os.path.exists(checkpoint_path):
        save_checkpoint(model, optimizer, scheduler, epoch, save_dir)
        print(f"Checkpoint saved for epoch {epoch + 1}!")

def validate(model, dataloader, loss_fct, device):
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

def save_checkpoint(model, optimizer, scheduler, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, optimizer, scheduler, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded. Resuming from epoch {epoch}.")
    return model, optimizer, scheduler, epoch