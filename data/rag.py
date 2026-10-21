
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state

    mask = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()

    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)

    return summed / counts


def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    return tokenizer, model


def encode_texts(texts, tokenizer, model):
    embeddings = []

    if len(texts) == 0:
        return np.empty((0, 384), dtype=np.float32)

    for i in range(0, len(texts), 16):
        batch = texts[i:i + 16]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        with torch.no_grad():
            output = model(**inputs)

        vectors = mean_pooling(
            output,
            inputs["attention_mask"]
        )

        vectors = torch.nn.functional.normalize(
            vectors, p=2, dim=1
        )

        embeddings.append(vectors.cpu().numpy())

    return np.vstack(embeddings)


def retrieve_answer(
    question, documents, embeddings, tokenizer, model, top_k=3
):
    if not question.strip():
        return []

    if len(documents) == 0 or len(embeddings) == 0:
        return []

    query_embedding = encode_texts(
        [question], tokenizer, model
    )

    scores = cosine_similarity(
        query_embedding, embeddings
    )[0]

    indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in indices:
         results.append({
        "index": int(idx),
        "document": documents[idx],
        "score": float(scores[idx])
        })

    return results
