import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizerFast

from model import SimpleBertForPreTraining
from data_prep import BertPretrainingDataset, load_and_split_wikitext_2, build_sentence_pairs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

EPOCHS = 3
LR = 5e-5
BATCH_SIZE = 8
VOCAB_SIZE = 30522
MAX_SEQ_LENGTH = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
NUM_HEADS = 4
INTERMEDIATE_SIZE = 512
DROPOUT = 0.1


tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased") # tokenizer


print("Loading and preparing WikiText-2 dataset...")
docs = load_and_split_wikitext_2() # from data_prep.py
pairs = build_sentence_pairs(docs, negative_ratio=1) # from data_prep.py

# Create dataset & dataloader
print(len(pairs))
dataset = BertPretrainingDataset(pairs[:1000], tokenizer, max_seq_length=MAX_SEQ_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


model = SimpleBertForPreTraining(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=MAX_SEQ_LENGTH,
    type_vocab_size=2,
    dropout=DROPOUT
).to(device)


mlm_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
nsp_loss_fn = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR)


print("Starting training")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        mlm_labels = batch["mlm_labels"].to(device)
        nsp_labels = batch["nsp_label"].to(device)

        optimizer.zero_grad()

        # Forward pass
        prediction_scores, seq_relationship_scores = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        # MLM loss
        mlm_loss = mlm_loss_fn(
            prediction_scores.view(-1, VOCAB_SIZE),
            mlm_labels.view(-1)
        )

        # NSP loss
        nsp_loss = nsp_loss_fn(seq_relationship_scores, nsp_labels)

        # Total loss
        loss = mlm_loss + nsp_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} completed, Avg loss: {avg_loss:.4f}")

print("Training Finished Successfully!")



