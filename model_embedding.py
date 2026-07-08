import jax
import jax.numpy as jnp

class ModelEmbedding:
    def __init__(self, src_vocab_size : int, tgt_vocab_size : int, embed_size : int, key : jax.random.key):
        src_key, tgt_key = jax.random.split(key)

        self._embed_size = embed_size

        self.source = jax.random.normal(
            src_key, (src_vocab_size, embed_size)
        ) * 0.01

        self.target = jax.random.normal(
            tgt_key, (tgt_vocab_size, embed_size)
        ) * 0.01

    def embed_size(self):
        return self._embed_size
    
    def all_src_embeddings(self):
        return self.source
    
    def all_tgt_embeddings(self):
        return self.target

    def source_embed(self, source_ids):
        return self.source[source_ids]

    def target_embed(self, target_ids):
        return self.target[target_ids]