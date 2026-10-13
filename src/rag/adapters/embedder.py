"""Embedder: turns text into unit-length vectors whose dot products measure similarity.

Every Embedder has the same port, so the Pipeline never knows which model it is using:

    model_id           recorded in the Index, which refuses Queries from any other model
    dimension          length of each vector
    max_input_tokens   longest text the model reads, special tokens not counted; anything
                       longer would be cut off
    embed_documents(texts) -> matrix with one unit vector per text
    embed_query(text)      -> one unit vector
"""


class BgeSmallEmbedder:
    """BAAI/bge-small-en-v1.5, run by hand: tokenize, forward pass, first-token pooling, L2-normalize."""

    model_id = "BAAI/bge-small-en-v1.5"
    dimension = 384
    # bge reads 512 positions, and its [CLS] and [SEP] markers always take two of them.
    max_input_tokens = 510

    # bge was trained with this instruction in front of short search queries (and nothing in
    # front of Chunks), so a Query and the Chunk that answers it land close together.
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, batch_size=16):
        # Imported here, not at the top, so code that never builds a real model stays fast.
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).eval()

    def embed_documents(self, texts):
        return self._embed(list(texts))

    def embed_query(self, text):
        return self._embed([self.QUERY_INSTRUCTION + text])[0]

    def _embed(self, texts):
        torch = self._torch
        batches = []
        for start in range(0, len(texts), self.batch_size):
            # 1. Tokenize: text -> token ids, padded to the longest text in the batch. The
            #    attention mask tells the model which positions are padding to ignore.
            inputs = self.tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens + 2,  # the text plus [CLS] and [SEP]
                return_tensors="pt",
            )
            # 2. Forward pass: one contextual vector per token, shape (batch, tokens, 384).
            with torch.inference_mode():
                token_vectors = self.model(**inputs).last_hidden_state
            # 3. Pool: bge is trained so the first token ([CLS]) summarises the whole text,
            #    so that one token's vector stands for the text.
            pooled = token_vectors[:, 0]
            # 4. L2-normalize: unit length makes the dot product equal the cosine similarity.
            batches.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
        return torch.cat(batches).numpy()


class Qwen3Embedder:
    """Qwen/Qwen3-Embedding-0.6B, run by hand: tokenize, forward pass, last-real-token pooling, L2-normalize."""

    model_id = "Qwen/Qwen3-Embedding-0.6B"
    dimension = 1024
    max_input_tokens = 32768

    # Qwen3 was trained with this instruction in front of search queries (and nothing in
    # front of Chunks), so a Query and the Chunk that answers it land close together.
    QUERY_INSTRUCTION = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

    def __init__(self, batch_size=8):
        # Imported here, not at the top, so code that never builds a real model stays fast.
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="right")
        self.model = AutoModel.from_pretrained(self.model_id).eval()

    def embed_documents(self, texts):
        return self._embed(list(texts))

    def embed_query(self, text):
        return self._embed([self.QUERY_INSTRUCTION + text])[0]

    def _pool(self, token_vectors, attention_mask):
        """Pool on the last real token of each sequence, ignoring padding tokens.

        bge-small uses first-token ([CLS]) pooling; Qwen3 is a causal decoder-only architecture
        where information accumulates into the final token. In a padded batch:
        - If padded on the left or unpadded, the last token is always at index -1.
        - If padded on the right, the index of the last real token is the count of
          non-padding tokens minus one (attention_mask.sum(dim=1) - 1).
        """
        torch = self._torch
        if attention_mask[:, -1].sum() == attention_mask.shape[0]:
            return token_vectors[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(token_vectors.shape[0], device=token_vectors.device)
        return token_vectors[batch_indices, sequence_lengths]

    def _embed(self, texts):
        torch = self._torch
        batches = []
        for start in range(0, len(texts), self.batch_size):
            # 1. Tokenize: text -> token ids, padded to the longest text in the batch.
            inputs = self.tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
                return_tensors="pt",
            )
            # 2. Forward pass: one contextual vector per token, shape (batch, tokens, 1024).
            with torch.inference_mode():
                token_vectors = self.model(**inputs).last_hidden_state
            # 3. Pool on the last real token.
            pooled = self._pool(token_vectors, inputs["attention_mask"]).float()
            # 4. L2-normalize: unit length makes the dot product equal the cosine similarity.
            batches.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
        return torch.cat(batches).numpy()

