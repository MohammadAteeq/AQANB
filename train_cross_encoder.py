import os
import json
import math
import hashlib
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from rank_bm25 import BM25Okapi
import wandb


class Config:
    """Centralized configuration for the training script."""
    # Model and training settings
    MODEL_NAME = "facebook/mcontriever"
    EMBEDDING_PATH = "./embeddings/embeddings_dict.json"
    DATA_FOLDER = "./narrativeqa-ar/passage_events_qa_10_100"
    BATCH_SIZE_TRAIN = 4
    BATCH_SIZE_VAL = 16
    MAX_LENGTH = 512
    NEGATIVE_SAMPLES_TRAIN = 8
    NEGATIVE_SAMPLES_VAL = 4
    LEARNING_RATE = 1e-6
    MAX_EPOCHS = 2
    GRADIENT_CLIP_VAL = 1.0
    PRECISION = "16-mixed"  # Mixed precision training
    DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
    WANDB_PROJECT = "retriever-training"
    WANDB_RUN_NAME = "cross-encoder-run"
    WANDB_API_KEY = ""
    CHECKPOINT_DIR = "./"
    CHECKPOINT_FILENAME = "best_checkpoint"
    MONITOR_METRIC = "val_recall_at_k"
    MONITOR_MODE = "max"


def compute_recall_at_k(similarity_scores, labels, k=1):
    """
    Compute Recall@K.
    Args:
        similarity_scores: Tensor of similarity scores (batch_size, num_candidates)
        labels: Tensor of ground-truth indices (batch_size)
        k: Top-K predictions to consider

    Returns:
        recall_at_k: Recall@K value
    """
    _, top_k_indices = similarity_scores.topk(k, dim=-1)
    correct = (labels.view(-1, 1) == top_k_indices).any(dim=-1)
    recall_at_k = correct.float().mean().item()
    return recall_at_k


def mean_pooling(hidden_states, attention_mask):
    """
    Compute the mean of the unmasked tokens.
    Args:
        hidden_states: Tensor of shape [batch_size, seq_length, hidden_dim]
        attention_mask: Tensor of shape [batch_size, seq_length]

    Returns:
        mean_pooled: Tensor of shape [batch_size, hidden_dim]
    """
    mask = attention_mask.unsqueeze(-1)
    masked_hidden_states = hidden_states * mask
    sum_embeddings = masked_hidden_states.sum(dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1)
    mean_pooled = sum_embeddings / token_counts
    return mean_pooled


class RetrieverDataset(Dataset):
    """Custom dataset for training the retriever model."""
    def __init__(self, file_list, tokenizer, max_length=Config.MAX_LENGTH, negative_samples=Config.NEGATIVE_SAMPLES_TRAIN):
        self.file_list = file_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.negative_samples = negative_samples
        self.data, self.bm25_retrievers = self._load_all_files_and_build_bm25()
        with open(Config.EMBEDDING_PATH, "r") as f:
            self.embeddings_dict = json.load(f)

    def _load_all_files_and_build_bm25(self):
        """Load files, tokenize passages, and build BM25 retrievers."""
        data = []
        bm25_retrievers = {}
        for file_path in self.file_list:
            df = pd.read_excel(file_path)
            passages = df["passage"].tolist()
            questions = df["question"].tolist()
            tokenized_passages = [p.split() for p in passages]
            bm25_retriever = BM25Okapi(tokenized_passages)
            file_name = os.path.basename(file_path)
            bm25_retrievers[file_name] = bm25_retriever
            for idx, (question, passage) in enumerate(zip(questions, passages)):
                data.append({
                    "file_name": file_name,
                    "question": question,
                    "passage": passage,
                    "position": idx,
                    "file_passages": passages,
                    "file_questions": questions,
                    "bm25_retriever": bm25_retriever,
                })
        return data, bm25_retrievers

    def hash_text(self, text):
        """Generate a SHA256 hash for a given text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_hard_negatives_with_scores(self, question, file_passages, bm25_retriever, position):
        """Retrieve hard negatives using BM25, distance weighting, and embedding similarity scores."""
        weights = {"bm25": 0.0, "distance": 0.5, "embedding": 0.5}
        query_tokens = question.split()
        bm25_scores = bm25_retriever.get_scores(query_tokens)
        min_score, max_score = min(bm25_scores), max(bm25_scores)
        normalized_bm25_scores = [(s - min_score) / (max_score - min_score) for s in bm25_scores]
        distances = [abs(i - position) for i in range(len(file_passages))]
        alpha = 0.5
        distance_weights = [math.exp(-alpha * d) for d in distances]
        question_hash = self.hash_text(question)
        question_embedding = np.array(self.embeddings_dict.get(question_hash, []))
        embedding_scores = []
        for passage in file_passages:
            passage_hash = self.hash_text(passage)
            passage_embedding = np.array(self.embeddings_dict.get(passage_hash, []))
            if question_embedding.size > 0 and passage_embedding.size > 0:
                embedding_score = np.dot(question_embedding, passage_embedding) / (
                    np.linalg.norm(question_embedding) * np.linalg.norm(passage_embedding)
                )
            else:
                embedding_score = 0.0
            embedding_scores.append(embedding_score)
        combined_scores = [
            weights["bm25"] * bm25_score +
            weights["distance"] * distance_weight +
            weights["embedding"] * embedding_score
            for bm25_score, distance_weight, embedding_score in zip(normalized_bm25_scores, distance_weights, embedding_scores)
        ]
        passage_scores = [(i, score) for i, score in enumerate(combined_scores) if i != position]
        hard_negatives = sorted(passage_scores, key=lambda x: x[1], reverse=True)[:self.negative_samples]
        hard_negative_passages = [file_passages[i] for i, _ in hard_negatives]
        hard_negative_scores = [score for _, score in hard_negatives]
        return hard_negative_passages, hard_negative_scores

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        question = entry["question"]
        positive_passage = entry["passage"]
        position = entry["position"]
        file_passages = entry["file_passages"]
        bm25_retriever = entry["bm25_retriever"]
        negative_passages, negative_scores = self._get_hard_negatives_with_scores(
            question, file_passages, bm25_retriever, position
        )
        positive_input = f"{question} {positive_passage}"
        negative_inputs = [f"{question} {neg}" for neg in negative_passages]
        positive_encoding = self.tokenizer(
            positive_input, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        negative_encodings = self.tokenizer(
            negative_inputs, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = torch.tensor([1.0] + negative_scores, dtype=torch.float32)
        return {
            "positive_input": {key: val.squeeze(0) for key, val in positive_encoding.items()},
            "negative_inputs": {key: val for key, val in negative_encodings.items()},
            "labels": labels,
        }


class RetrieverModel(pl.LightningModule):
    """PyTorch Lightning module for the retriever model."""
    def __init__(self, model_name=Config.MODEL_NAME, lr=Config.LEARNING_RATE):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.fc_layer = torch.nn.Linear(self.encoder.config.hidden_size, 1)
        self.lr = lr

    def forward(self, positive_input, negative_inputs):
        positive_outputs = self.encoder(**positive_input)
        positive_pooled_output = mean_pooling(positive_outputs.last_hidden_state, positive_input["attention_mask"])
        positive_score = torch.sigmoid(self.fc_layer(positive_pooled_output)).squeeze(-1)
        negative_scores = []
        for i in range(negative_inputs["input_ids"].size(1)):
            neg_input = {key: negative_inputs[key][:, i, :] for key in negative_inputs}
            neg_outputs = self.encoder(**neg_input)
            neg_pooled_output = mean_pooling(neg_outputs.last_hidden_state, neg_input["attention_mask"])
            neg_score = torch.sigmoid(self.fc_layer(neg_pooled_output)).squeeze(-1)
            negative_scores.append(neg_score)
        negative_scores = torch.stack(negative_scores, dim=1)
        return positive_score, negative_scores

    def training_step(self, batch, batch_idx):
        positive_score, negative_scores = self.forward(batch["positive_input"], batch["negative_inputs"])
        all_scores = torch.cat([positive_score.unsqueeze(1), negative_scores], dim=1)
        loss = F.mse_loss(all_scores, batch["labels"].to(self.device))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        positive_score, negative_scores = self.forward(batch["positive_input"], batch["negative_inputs"])
        all_scores = torch.cat([positive_score.unsqueeze(1), negative_scores], dim=1)
        loss = F.mse_loss(all_scores, batch["labels"].to(self.device))
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        recall_at_k = compute_recall_at_k(all_scores, torch.zeros(batch["labels"].size(0), dtype=torch.long).to(self.device))
        self.log("val_recall_at_k", recall_at_k, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        positive_score, negative_scores = self.forward(batch["positive_input"], batch["negative_inputs"])
        all_scores = torch.cat([positive_score.unsqueeze(1), negative_scores], dim=1)
        loss = F.mse_loss(all_scores, batch["labels"].to(self.device))
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        recall_at_k = compute_recall_at_k(all_scores, torch.zeros(batch["labels"].size(0), dtype=torch.long).to(self.device))
        self.log("test_recall_at_k", recall_at_k, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)


def main():
    """Main function to train the retriever model."""
    # Prepare data
    file_paths = [os.path.join(Config.DATA_FOLDER, file) for file in os.listdir(Config.DATA_FOLDER) if file.endswith(".xlsx")]
    train_files, val_files = train_test_split(file_paths, test_size=0.1, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    train_dataset = RetrieverDataset(train_files, tokenizer, negative_samples=Config.NEGATIVE_SAMPLES_TRAIN)
    val_dataset = RetrieverDataset(val_files, tokenizer, negative_samples=Config.NEGATIVE_SAMPLES_VAL)
    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE_TRAIN, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE_VAL, shuffle=False)

    # Initialize model and trainer
    model = RetrieverModel()
    wandb.login(key=Config.WANDB_API_KEY)
    wandb_logger = WandbLogger(project=Config.WANDB_PROJECT, name=Config.WANDB_RUN_NAME, log_model=False)
    checkpoint_callback = ModelCheckpoint(
        dirpath=Config.CHECKPOINT_DIR,
        filename=Config.CHECKPOINT_FILENAME,
        monitor=Config.MONITOR_METRIC,
        mode=Config.MONITOR_MODE,
        save_top_k=1,
        save_last=False,
    )
    trainer = pl.Trainer(
        precision=Config.PRECISION,
        max_epochs=Config.MAX_EPOCHS,
        accelerator=Config.DEVICE,
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=1,
        logger=wandb_logger,
        gradient_clip_val=Config.GRADIENT_CLIP_VAL,
        callbacks=[checkpoint_callback],
    )

    # Train and validate the model
    trainer.fit(model, train_dataloader, val_dataloader)

    # Save the model
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"{Config.CHECKPOINT_FILENAME}.ckpt")
    checkpoint = torch.load(checkpoint_path)
    new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if "question_encoder" not in k}
    checkpoint["state_dict"] = new_state_dict
    output_bin_path = "pytorch_model.bin"
    torch.save(checkpoint["state_dict"], output_bin_path)

    # Upload to wandb
    run = wandb.init(project="EnglishNarrativeRankerReader", name="Uploading model")
    artifact = wandb.Artifact("model", type="ranker_reader_models")
    artifact.add_file(output_bin_path)
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()