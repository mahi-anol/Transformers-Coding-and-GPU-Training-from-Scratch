"""
Inference module for the Transformer model.
Contains utilities and functions for translation inference on single and batch inputs.
"""

import torch


def causal_mask_generator(size):
    """
    Generate a causal mask to prevent attention to future positions.
    
    Args:
        size: Sequence length
    
    Returns:
        Causal mask of shape (1, size, size)
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


def encoder_preprocess(text, tokenizer_en, max_seq_len=None):
    """
    Preprocess a single English sentence for encoding.
    
    Args:
        text: Input English sentence
        tokenizer_en: English tokenizer
        max_seq_len: Maximum sequence length (optional)
    
    Returns:
        Tensor of shape (seq_len,) with encoded tokens
    """
    encoded_english_sentence = tokenizer_en.encode(text).ids
    processed_text = torch.cat(
        [
            torch.tensor(encoded_english_sentence, dtype=torch.int64),
            torch.tensor([tokenizer_en.token_to_id('[EOS]')], dtype=torch.int64),
        ]
    )
    
    if max_seq_len is not None and len(processed_text) > max_seq_len:
        processed_text = processed_text[:max_seq_len]
    
    return processed_text


def encoder_preprocess_batch(texts, tokenizer_en, max_seq_len=None):
    """
    Preprocess a batch of English sentences for encoding.
    
    Args:
        texts: List of English sentences
        tokenizer_en: English tokenizer
        max_seq_len: Maximum sequence length (optional)
    
    Returns:
        List of padded tensors of shape (max_text_len,)
    """
    tokenized_texts = []
    ideal_seq_len = float('-inf')
    
    for text in texts:
        tokenized_text = tokenizer_en.encode(text).ids
        ideal_seq_len = max(len(tokenized_text) + 1, ideal_seq_len)  # +1 for EOS token
        tokenized_texts.append(tokenized_text)
    
    padded_tokenized_texts = []
    
    for tokenized_text in tokenized_texts:
        padded_text = torch.cat(
            [
                torch.tensor(tokenized_text, dtype=torch.int64),
                torch.tensor([tokenizer_en.token_to_id('[EOS]')], dtype=torch.int64),
                torch.full(
                    (ideal_seq_len - len(tokenized_text) - 1,),
                    tokenizer_en.token_to_id('[PAD]')
                )
            ]
        )
        padded_tokenized_texts.append(padded_text)
    
    return padded_tokenized_texts


def generate(model, english_text, tokenizer_en, tokenizer_fr, max_seq_len=432, device='cpu'):
    """
    Generate French translation for a single English sentence.
    
    Args:
        model: Transformer model
        english_text: English sentence to translate
        tokenizer_en: English tokenizer
        tokenizer_fr: French tokenizer
        max_seq_len: Maximum sequence length for generation
        device: Device to run inference on
    
    Returns:
        Generated French translation string
    """
    model.eval()
    
    with torch.inference_mode():
        # Encode input
        processed_encoded_english_sentence = encoder_preprocess(
            english_text, tokenizer_en
        ).unsqueeze(0).to(device)
        
        # Create encoder mask
        encoder_mask = (
            (processed_encoded_english_sentence != tokenizer_en.token_to_id('[PAD]'))
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
        )
        
        # Encode
        encoder_output = model.encode(processed_encoded_english_sentence, encoder_mask)
        
        # Initialize decoder input with SOS token
        decoder_input = torch.tensor(
            [tokenizer_fr.token_to_id("[SOS]")],
            dtype=torch.int64
        ).unsqueeze(0).to(device)
        
        # Generate tokens step by step
        for _ in range(max_seq_len):
            if decoder_input.shape[1] >= max_seq_len:
                break
            
            # Create decoder mask
            decoder_mask = (
                (decoder_input != tokenizer_fr.token_to_id("[PAD]"))
                .unsqueeze(0)
                .unsqueeze(0)
            ) & causal_mask_generator(decoder_input.shape[1])
            decoder_mask = decoder_mask.to(device)
            
            # Decode
            decoder_output = model.decode(
                decoder_input, encoder_output, decoder_mask, encoder_mask
            )
            
            # Get next token
            predicted_word = torch.argmax(decoder_output[:, -1, :], dim=-1)
            
            # Append to decoder input
            decoder_input = torch.cat(
                [
                    decoder_input,
                    predicted_word.unsqueeze(0)
                ],
                dim=-1
            )
            
            # Stop if EOS or PAD token
            if (predicted_word == tokenizer_fr.token_to_id("[EOS]") or
                predicted_word == tokenizer_fr.token_to_id("[PAD]")):
                break
        
        # Decode tokens to text
        predicted_sentence = tokenizer_fr.decode(
            decoder_input.squeeze(0).tolist(),
            skip_special_tokens=False
        )
        
        return predicted_sentence


def generate_batch(model, english_texts, tokenizer_en, tokenizer_fr, max_seq_len=432, device='cpu'):
    """
    Generate French translations for a batch of English sentences.
    
    Args:
        model: Transformer model
        english_texts: List of English sentences to translate
        tokenizer_en: English tokenizer
        tokenizer_fr: French tokenizer
        max_seq_len: Maximum sequence length for generation
        device: Device to run inference on
    
    Returns:
        List of generated French translation strings
    """
    model.eval()
    
    with torch.inference_mode():
        # Preprocess batch
        processed_encoder_texts = encoder_preprocess_batch(english_texts, tokenizer_en)
        
        # Create encoder masks
        processed_encoder_masks = [
            (
                (processed_encoder_text != tokenizer_en.token_to_id('[PAD]'))
                .unsqueeze(0)
                .unsqueeze(0)
                .int()
            )
            for processed_encoder_text in processed_encoder_texts
        ]
        
        # Stack batch
        encoder_input_batch = torch.stack(processed_encoder_texts).to(device)
        encoder_mask_batch = torch.stack(processed_encoder_masks).to(device)
        
        # Encode
        encoder_output = model.encode(encoder_input_batch, encoder_mask_batch)
        
        # Initialize decoder input with SOS token for each sample in batch
        decoder_input_batch = torch.tensor(
            [tokenizer_fr.token_to_id("[SOS]") for _ in range(len(english_texts))],
            dtype=torch.int64
        ).unsqueeze(1).to(device)  # Shape: (batch_size, 1)
        
        # Generate tokens step by step
        for _ in range(max_seq_len):
            if decoder_input_batch.shape[1] >= max_seq_len:
                break
            
            # Create decoder masks for batch
            decoder_masks = [
                (
                    (decoder_input != tokenizer_fr.token_to_id("[PAD]"))
                    .unsqueeze(0)
                    .unsqueeze(0)
                ) & causal_mask_generator(decoder_input.shape[0])
                for decoder_input in decoder_input_batch
            ]
            decoder_mask_batch = torch.stack(decoder_masks).to(device)
            
            # Decode
            decoder_output = model.decode(
                decoder_input_batch, encoder_output, decoder_mask_batch, encoder_mask_batch
            )
            
            # Get next tokens
            predicted_words = torch.argmax(decoder_output[:, -1, :], dim=-1, keepdim=True)
            
            # Append to decoder input
            decoder_input_batch = torch.cat(
                [decoder_input_batch, predicted_words],
                dim=-1
            )
            
            # Check if all samples have generated EOS or PAD
            all_finished = True
            for i in range(len(english_texts)):
                last_token = predicted_words[i, -1].item()
                if not (
                    last_token == tokenizer_fr.token_to_id("[EOS]") or
                    last_token == tokenizer_fr.token_to_id("[PAD]")
                ):
                    all_finished = False
                    break
            
            if all_finished:
                break
        
        # Decode batch
        predicted_sentences = tokenizer_fr.decode_batch(
            decoder_input_batch.tolist(),
            skip_special_tokens=False
        )
        
        return predicted_sentences
