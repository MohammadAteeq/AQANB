import os
import json
import math
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from rank_bm25 import BM25Okapi
import wandb

# Configuration Class
class Config:
    # Paths
    EMBEDDINGS_PATH = './embeddings/embeddings_dict.json'
    DATA_FOLDER_PATH = './narrativeqa-ar/passage_events_qa_10_100'
    OUTPUT_MODEL_PATH = 'pytorch_model.bin'
    CHECKPOINT_PATH = './best_checkpoint.ckpt'

    # Model and Training Config
    MODEL_NAME = 'facebook/mcontriever'
    RUN_NAME = ""
    USE_CONTRASTIVE_LOSS = False
    SHARED_CONTEXT_QUESTION_MODEL = True
    LEARNING_RATE = 1e-6
    MAX_LENGTH = 400
    NEGATIVE_SAMPLES_TRAIN = 8
    NEGATIVE_SAMPLES_VAL = 4
    BATCH_SIZE_TRAIN = 4
    BATCH_SIZE_VAL = 16
    MAX_EPOCHS = 2
    PRECISION = '16-mixed'
    GRADIENT_CLIP_VAL = 1.0

    # WandB Config
    WANDB_API_KEY = "____"
    WANDB_PROJECT = "retriever-training"
    WANDB_RUN_NAME = RUN_NAME


# Helper Functions
def compute_recall_at_k(similarity_scores, labels, k=1):
    _, top_k_indices = similarity_scores.topk(k, dim=-1)
    correct = (labels.view(-1, 1) == top_k_indices).any(dim=-1)
    recall_at_k = correct.float().mean().item()
    return recall_at_k


def mean_pooling(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1)
    masked_hidden_states = hidden_states * mask
    sum_embeddings = masked_hidden_states.sum(dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1)
    mean_pooled = sum_embeddings / token_counts
    return mean_pooled


def contrastive_loss(question_embedding, positive_embedding, negative_embeddings, margin=1.0):
    positive_similarity = F.cosine_similarity(question_embedding, positive_embedding, dim=-1)
    batch_size, num_negatives, embedding_dim = negative_embeddings.size()
    question_embedding_expanded = question_embedding.unsqueeze(1).expand(-1, num_negatives, -1)
    negative_similarities = F.cosine_similarity(
        question_embedding_expanded.reshape(-1, embedding_dim),
        negative_embeddings.reshape(-1, embedding_dim),
        dim=-1
    ).reshape(batch_size, num_negatives)
    positive_loss = (1 - positive_similarity).pow(2)
    negative_loss = torch.clamp(margin + negative_similarities, min=0).pow(2)
    loss = positive_loss.mean() + negative_loss.mean()
    return loss


# Dataset Class
class RetrieverDataset(Dataset):
    def __init__(self, file_list, tokenizer, max_length, negative_samples, embeddings_dict):
        self.file_list = file_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.negative_samples = negative_samples
        self.embeddings_dict = embeddings_dict
        self.data, self.bm25_retrievers = self._load_all_files_and_build_bm25()

    def _load_all_files_and_build_bm25(self):
        data = []
        bm25_retrievers = {}
        for file_path in self.file_list:
            df = pd.read_excel(file_path)
            passages = df['passage'].tolist()
            questions = df['question'].tolist()
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
                    "bm25_retriever": bm25_retriever
                })
        return data, bm25_retrievers

    def hash_text(self, text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _get_negatives_with_scores(self, question, file_passages, bm25_retriever, position):
        weights = {'bm25': 0.0, 'distance': 1.0, 'embedding': 0.0}
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
            weights['bm25'] * bm25_score +
            weights['distance'] * distance_weight +
            weights['embedding'] * embedding_score
            for bm25_score, distance_weight, embedding_score in zip(normalized_bm25_scores, distance_weights, embedding_scores)
        ]
        passage_scores = [(i, score) for i, score in enumerate(combined_scores) if i != position]
        negatives = sorted(passage_scores, key=lambda x: x[1], reverse=True)[:self.negative_samples]
        negative_passages = [file_passages[i] for i, _ in negatives]
        negative_scores = [score for _, score in negatives]
        return negative_passages, negative_scores

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        question = entry["question"]
        positive_passage = entry["passage"]
        position = entry["position"]
        file_passages = entry["file_passages"]
        bm25_retriever = entry["bm25_retriever"]
        negative_passages, negative_scores = self._get_negatives_with_scores(
            question, file_passages, bm25_retriever, position
        )
        question_encoding = self.tokenizer(
            question, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        positive_passage_encoding = self.tokenizer(
            positive_passage, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        negative_passage_encodings = self.tokenizer(
            negative_passages, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = torch.tensor([1.0] + negative_scores, dtype=torch.float32)
        return {
            "question_input": {key: val.squeeze(0) for key, val in question_encoding.items()},
            "passage_input": {
                "positive": {key: val.squeeze(0) for key, val in positive_passage_encoding.items()},
                "negative": {key: val for key, val in negative_passage_encodings.items()},
            },
            "labels": labels
        }


# Model Class
class BiEncoderRetriever(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.context_encoder = AutoModel.from_pretrained(model_name)
        self.question_encoder = self.context_encoder if Config.SHARED_CONTEXT_QUESTION_MODEL else AutoModel.from_pretrained(model_name)
        self.lr = lr

    def forward(self, question_input, passage_input):
        question_outputs = self.question_encoder(**question_input)
        question_embedding = mean_pooling(question_outputs.last_hidden_state, question_input["attention_mask"])
        positive_outputs = self.context_encoder(**passage_input["positive"])
        positive_embedding = mean_pooling(positive_outputs.last_hidden_state, passage_input["positive"]["attention_mask"])
        negative_input = {key: val.view(-1, val.size(-1)) for key, val in passage_input["negative"].items()}
        negative_outputs = self.context_encoder(**negative_input)
        negative_embeddings = mean_pooling(negative_outputs.last_hidden_state, negative_input["attention_mask"])
        batch_size = passage_input["negative"]["input_ids"].size(0)
        num_negatives = passage_input["negative"]["input_ids"].size(1)
        negative_embeddings = negative_embeddings.view(batch_size, num_negatives, -1)
        return question_embedding, positive_embedding, negative_embeddings

    def compute_similarity(self, question_embedding, positive_embedding, negative_embeddings):
        all_embeddings = torch.cat([positive_embedding.unsqueeze(1), negative_embeddings], dim=1)
        similarity_scores = F.cosine_similarity(question_embedding.unsqueeze(1), all_embeddings, dim=-1)
        return similarity_scores

    def training_step(self, batch, batch_idx):
        question_embedding, positive_embedding, negative_embeddings = self.forward(batch["question_input"], batch["passage_input"])
        if Config.USE_CONTRASTIVE_LOSS:
            loss = contrastive_loss(question_embedding, positive_embedding, negative_embeddings)
        else:
            similarity_scores = self.compute_similarity(question_embedding, positive_embedding, negative_embeddings)
            loss = F.mse_loss(similarity_scores, batch["labels"].to(self.device))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        question_embedding, positive_embedding, negative_embeddings = self.forward(batch["question_input"], batch["passage_input"])
        similarity_scores = self.compute_similarity(question_embedding, positive_embedding, negative_embeddings)
        if Config.USE_CONTRASTIVE_LOSS:
            loss = contrastive_loss(question_embedding, positive_embedding, negative_embeddings)
        else:
            loss = F.mse_loss(similarity_scores, batch["labels"].to(self.device))
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        recall_at_k = compute_recall_at_k(similarity_scores, torch.zeros(batch["labels"].size(0), dtype=torch.long).to(self.device))
        self.log("val_recall_at_k", recall_at_k, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        question_embedding, positive_embedding, negative_embeddings = self.forward(batch["question_input"], batch["passage_input"])
        similarity_scores = self.compute_similarity(question_embedding, positive_embedding, negative_embeddings)
        if Config.USE_CONTRASTIVE_LOSS:
            loss = contrastive_loss(question_embedding, positive_embedding, negative_embeddings)
        else:
            loss = F.mse_loss(similarity_scores, batch["labels"].to(self.device))
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        recall_at_k = compute_recall_at_k(similarity_scores, torch.zeros(batch["labels"].size(0), dtype=torch.long).to(self.device))
        self.log("test_recall_at_k", recall_at_k, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        return optimizer


# Main Execution
if __name__ == "__main__":
    # Load data
    file_paths = [os.path.join(Config.DATA_FOLDER_PATH, file) for file in os.listdir(Config.DATA_FOLDER_PATH) if file.endswith(".xlsx")]
    train_files, val_files = train_test_split(file_paths, test_size=0.1, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    with open(Config.EMBEDDINGS_PATH, 'r') as f:
        embeddings_dict = json.load(f)
    train_dataset = RetrieverDataset(train_files, tokenizer, Config.MAX_LENGTH, Config.NEGATIVE_SAMPLES_TRAIN, embeddings_dict)
    val_dataset = RetrieverDataset(val_files, tokenizer, Config.MAX_LENGTH, Config.NEGATIVE_SAMPLES_VAL, embeddings_dict)
    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE_TRAIN, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE_VAL, shuffle=False)

    # Initialize WandB
    wandb.login(key=Config.WANDB_API_KEY)
    wandb_logger = WandbLogger(project=Config.WANDB_PROJECT, name=Config.WANDB_RUN_NAME, log_model=False)
    checkpoint_callback = ModelCheckpoint(dirpath="./", filename="best_checkpoint", monitor="val_recall_at_k", mode="max", save_top_k=1, save_last=False)

    # Initialize and train model
    model = BiEncoderRetriever(Config.MODEL_NAME, Config.LEARNING_RATE)
    trainer = pl.Trainer(
        precision=Config.PRECISION,
        max_epochs=Config.MAX_EPOCHS,
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=1,
        logger=wandb_logger,
        gradient_clip_val=Config.GRADIENT_CLIP_VAL,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    # Convert checkpoint to .bin format
    checkpoint = torch.load(Config.CHECKPOINT_PATH)
    new_state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        if 'question_encoder' in k and Config.SHARED_CONTEXT_QUESTION_MODEL:
            continue
        new_state_dict[k] = v
    checkpoint['state_dict'] = new_state_dict
    torch.save(checkpoint['state_dict'], Config.OUTPUT_MODEL_PATH)

    # Upload model to WandB
    run = wandb.init(project='EnglishNarrativeRankerReader', name='Uploading model')
    artifact = wandb.Artifact('model', type='ranker_reader_models')
    artifact.add_file(Config.OUTPUT_MODEL_PATH)
    run.log_artifact(artifact)
    run.finish()