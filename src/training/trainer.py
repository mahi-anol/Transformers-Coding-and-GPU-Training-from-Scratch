"""
Training module for the Transformer model.
Contains the training loop, validation, and related utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


@torch.no_grad()
def get_prediction_accuracy(predictions, targets, pad_token_id):
    """
    Calculate prediction accuracy excluding padding tokens.
    
    Args:
        predictions: Model predictions of shape (batch, seq_len, vocab_size)
        targets: Target tokens of shape (batch, seq_len)
        pad_token_id: Token ID for padding
    
    Returns:
        Accuracy as a float tensor
    """
    prediction_tokens = predictions.argmax(dim=-1)
    mask = targets != pad_token_id
    correct = (prediction_tokens == targets) & mask
    acc = correct.sum().float() / mask.sum().float()
    return acc


def train_epoch(model, train_loader, loss_fn, optimizer, pad_token_id, device='cpu'):
    """
    Train the model for one epoch.
    
    Args:
        model: Transformer model
        train_loader: Training data loader
        loss_fn: Loss function
        optimizer: Optimizer
        pad_token_id: Padding token ID
        device: Device to train on
    
    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    
    train_loader_tqdm = tqdm(train_loader, desc="Training Epoch", leave=False)
    for i, batch in enumerate(train_loader_tqdm):
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        target_output = batch['target_output'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)
        
        # Forward pass
        decoder_output = model(decoder_input, encoder_input, decoder_mask, encoder_mask)
        
        # Calculate accuracy
        acc = get_prediction_accuracy(decoder_output, target_output, pad_token_id)
        train_acc += acc
        
        # Calculate loss
        loss = loss_fn(
            decoder_output.reshape(-1, decoder_output.shape[-1]),
            target_output.reshape(-1)
        )
        train_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        train_loader_tqdm.set_postfix({
            f"batch_{i+1}_loss": loss.item(),
            f"batch_{i+1}_acc": acc.item()
        })
    
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    
    return train_loss, train_acc


def validate_epoch(model, val_loader, loss_fn, pad_token_id, device='cpu'):
    """
    Validate the model for one epoch.
    
    Args:
        model: Transformer model
        val_loader: Validation data loader
        loss_fn: Loss function
        pad_token_id: Padding token ID
        device: Device to validate on
    
    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.eval()
    validation_loss = 0.0
    validation_acc = 0.0
    
    with torch.no_grad():
        validation_loader_tqdm = tqdm(val_loader, desc="Validation Epoch", leave=False)
        for i, batch in enumerate(validation_loader_tqdm):
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            target_output = batch['target_output'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            
            # Forward pass
            decoder_output = model(decoder_input, encoder_input, decoder_mask, encoder_mask)
            
            # Calculate accuracy
            acc = get_prediction_accuracy(decoder_output, target_output, pad_token_id)
            validation_acc += acc
            
            # Calculate loss
            loss = loss_fn(
                decoder_output.reshape(-1, decoder_output.shape[-1]),
                target_output.reshape(-1)
            )
            validation_loss += loss.item()
            
            validation_loader_tqdm.set_postfix({
                f"batch_{i+1}_loss": loss.item(),
                f"batch_{i+1}_acc": acc.item()
            })
    
    validation_loss /= len(val_loader)
    validation_acc /= len(val_loader)
    
    return validation_loss, validation_acc


def train(model, train_loader, val_loader, config, tokenizer_config, device='cpu'):
    """
    Main training loop.
    
    Args:
        model: Transformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict with keys:
            - epochs: Number of epochs to train
            - optimizer_lr: Learning rate
            - optimizer_eps: Epsilon for Adam optimizer
            - label_smoothing: Label smoothing factor
        tokenizer_config: Tokenizer configuration with 'fr_pad_token_id'
        device: Device to train on
    
    Returns:
        Dictionary with training history
    """
    # Setup loss function and optimizer
    pad_token_id = tokenizer_config.get('fr_pad_token_id')
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=pad_token_id,
        label_smoothing=config.get('label_smoothing', 0.1)
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('optimizer_lr', 1e-4),
        eps=config.get('optimizer_eps', 1e-8)
    )
    
    epochs = config.get('epochs', 3)
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, pad_token_id, device
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, loss_fn, pad_token_id, device
        )
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"Epoch: {epoch+1} | "
              f"Train loss: {train_loss:.4f} | "
              f"Train acc: {train_acc:.4f} | "
              f"Validation loss: {val_loss:.4f} | "
              f"Validation acc: {val_acc:.4f}")
    
    return history


def save_model(model, path):
    """Save model weights to disk."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device='cpu'):
    """Load model weights from disk."""
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")
    return model
