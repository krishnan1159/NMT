import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.lax as jlax
from jax.typing import ArrayLike
from typing import List, Dict
from model_embedding import ModelEmbedding
from vocab import Tokenizer
from first_logger import FirstNLogger
from config import TrainConfig
import logging
import time
from functools import partial

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # change to DEBUG when you want rnn.py logs.
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(handler)

"""
    Trying to write purely in functional programming style. Think in terms of what is data and what are the transformations we will apply to the data.

    Plan(Basic RNN):
    ================

        Basic RNN (D, H, V):
        ====================
        D - dimension of the embedding vector
        H - dimension of the hidden vector
        V - Vocab Size
        Wₕₕ - (H x H)
        Wₕₓ - (H x D)
        Wₕᵧ - (V x H)


        Implementation Plan:
        ====================

        1. Have a train method which takes all the source, target sentences and ModelEmbedding class to embed the sentences.
        2. Define the encoder the step to process one single token. Using jax.lax.scan roll it for all tokens to generate encoded vector.
        3. Define the decoder step. In first step we will use the encoded vector. In further steps, previous state hidden vector. Use softmax to determin what is the predicted word.
            Use the predicted word in previous step as input in the current step.
        4. 

        Encoder Step:
        =============
        uₜ = xₜ . Wₕₓᵀ + hₜ₋₁ . Wₕₕᵀ + b
        hₜ = tanh(uₜ)

"""

def init_rnn_encoder_params(key, embed_size, hidden_size):
    k1, k2 = jax.random.split(key, 2)
    return {
        "Whx" : jax.random.normal(k1, (hidden_size, embed_size)) * 0.01,
        "Whh" : jax.random.normal(k2, (hidden_size, hidden_size)) * 0.01,
        "b"   : jnp.zeros(1)
    }

def init_rnn_decoder_params(key, embed_size, hidden_size, vocab_size):
    k1, k2, k3 = jax.random.split(key, 3)

    return {
        "Whx" : jax.random.normal(k1, (hidden_size, embed_size)) * 0.01,
        "Whh" : jax.random.normal(k2, (hidden_size, hidden_size)) * 0.01,
        "Why" : jax.random.normal(k3, (vocab_size, hidden_size)) * 0.01,
        "b"   : jnp.zeros(1)
    }

def init_params(src_embeddings : ArrayLike, tgt_embeddings: ArrayLike, embed_size : int, hidden_size : int, tgt_vocab_size : int):
    encoderRandomKey, decoderRandomKey = jax.random.split(jax.random.key(1), 2)
    rnn_encoder_params = init_rnn_encoder_params(encoderRandomKey, embed_size, hidden_size)
    rnn_decoder_params = init_rnn_decoder_params(decoderRandomKey, embed_size, hidden_size, tgt_vocab_size)

    embeddings = {
        "source" : src_embeddings,
        "target" : tgt_embeddings
    }

    params = {
        "embeddings" : embeddings,
        "encoder": rnn_encoder_params,
        "decoder": rnn_decoder_params
    }

    return params

def rnn_encoder_cell(params : dict, htminus : ArrayLike, xt_with_mask : ArrayLike): #htminus is carry

    xt, mask = xt_with_mask

    Whx, Whh, b = params["encoder"]["Whx"], params["encoder"]["Whh"], params["encoder"]["b"]

    unmasked_ht = jnn.tanh(jnp.dot(xt, Whx.T) + jnp.dot(htminus, Whh.T) + b)
    zeroes = jnp.zeros_like(unmasked_ht)
    ht = jnp.where(mask[:, None], unmasked_ht, htminus)
    hout = jnp.where(mask[:, None], unmasked_ht, zeroes)
    return ht, hout

def rnn_encoder_batch(rnn_params : Dict[str, ArrayLike], batch_src_embeddings : ArrayLike, batch_src_lengths : ArrayLike):

    # batch_src_embeddings: (B, T, D)
    batch_src_embeddings_t = jnp.swapaxes(batch_src_embeddings, 0, 1)
    # batch_src_embeddings_t: (T, B, D)

    mask = jnp.arange(batch_src_embeddings.shape[1]) < batch_src_lengths[:, None]
    hidden_size = rnn_params["encoder"]["Whh"].shape[0]
    h0 = jnp.zeros((batch_src_embeddings_t.shape[1], hidden_size))
        
    henc, encoder_outputs = jlax.scan(lambda carry, x: rnn_encoder_cell(rnn_params, carry, x), h0, (batch_src_embeddings_t, mask.T))
    
    return henc, encoder_outputs, mask

def rnn_decoder_cell(params : Dict[str, ArrayLike], htminus : ArrayLike, inp_t: ArrayLike):
    # In lax.scan terms: htminus is the carry, inp_t is the xs value.
    xt, next_tokens, mask = inp_t ## xt -> (B x embed_size), next_tokens -> (B, ) mask -> (B, )
    ## Get the params
    Whx, Whh, Why, b = params["decoder"]["Whx"], params["decoder"]["Whh"], params["decoder"]["Why"], params["decoder"]["b"]
    
    unmasked_ht = jnn.tanh(jnp.dot(xt, Whx.T) + jnp.dot(htminus, Whh.T) + b) # unmasked_ht -> (B, H)
    ht = jnp.where(mask[:, None], unmasked_ht, htminus) # ht -> (B, H)

    vt = jnp.dot(ht, Why.T) # vt will be (B, V)
    log_probs = jnn.log_softmax(vt) # log_probs will be (B, V). Each row will sum up to 1.
    token_loss = -log_probs[jnp.arange(vt.shape[0]), next_tokens]
    loss = jnp.sum(jnp.where(mask, token_loss, 0.0)) # loss represents loss per batch

    return ht, loss

def rnn_decoder_batch(rnn_decoder_params : Dict[str, ArrayLike], batch_tgt_embeddings : ArrayLike, batch_tgt_lengths : ArrayLike, batch_next_tokens : ArrayLike, henc : ArrayLike):
    # Shapes : batch_tgt_embeddings -> (B, T, D), batch_next_tokens -> (B, T), henc -> (B, H)

    ## lax.scan will roll over the leading axes. If we have (B, T, D), scan will roll over all tokens in a sentence at each step.
    ## We need swap the axes to (T, B, D) so at each step, we will process 1st token of all sentences and then 2nd token of all sentences and so on.
    batch_tgt_embeddings_t = batch_tgt_embeddings.swapaxes(0, 1)

    ## Creating mask array from sequence lengths. "jnp.arange()" will create array of size "(1, max(batch_tgt_lengths))" and "batch_tgt_lengths[:, None]" will change array to "(len(batch_tgt_lengths), 1)"
    ## When we use any arithmetic or logical operator, broadcasting rule comes into picture. This operation will create array of size (len(batch_tgt_lengths), max(batch_tgt_lengths))
    mask = jnp.arange(batch_tgt_embeddings.shape[1]) < batch_tgt_lengths[:, None]

    ## batch_tgt_embeddings_t is (T, B, D), mask.T is (T, B), batch_next_tokens.T is (T, B). lax.scan will roll over one time step at a time and compute the overall loss.
    _, loss = jax.lax.scan(
        lambda carry, xs : rnn_decoder_cell(rnn_decoder_params, carry, xs),
        henc, (batch_tgt_embeddings_t, batch_next_tokens.T, mask.T))
    
    return jnp.sum(loss)

def rnn_process_one_batch(rnn_params : Dict, batch_src_embeddings : ArrayLike, batch_src_lengths : ArrayLike, batch_tgt_embeddings : ArrayLike, batch_tgt_lengths : ArrayLike, batch_decoder_expected_tokens: ArrayLike):
    henc, _, _ = rnn_encoder_batch(rnn_params, batch_src_embeddings, batch_src_lengths)
    return rnn_decoder_batch(rnn_params, batch_tgt_embeddings, batch_tgt_lengths, batch_decoder_expected_tokens, henc)

def train_one_batch(rnn_params : Dict, batch_src_tokens : ArrayLike, batch_src_lengths : ArrayLike, batch_tgt_tokens : ArrayLike, batch_tgt_lengths : ArrayLike, batch_tgt_next_tokens : ArrayLike, lr : float):
    batch_src_embeddings  = rnn_params["embeddings"]["source"][batch_src_tokens]
    batch_tgt_embeddings  = rnn_params["embeddings"]["target"][batch_tgt_tokens]
    loss, grads = jax.value_and_grad(rnn_process_one_batch)(rnn_params, batch_src_embeddings, batch_src_lengths, batch_tgt_embeddings, batch_tgt_lengths, batch_tgt_next_tokens)
    rnn_params = jax.tree.map(lambda p, g: p - lr * g, rnn_params, grads)
    return loss, rnn_params

@partial(jax.jit, static_argnames=("batch_size",))
def train_one_epoch(rnn_params : Dict, src_sents_tokens : ArrayLike, tgt_sents_tokens : ArrayLike, src_sents_lengths: ArrayLike, tgt_sents_lengths: ArrayLike, batch_size: int):
    num_sents = src_sents_tokens.shape[0]
    total_loss = 0.0
    total_tokens = 0
    lr = 1e-4

    for start in range(0, num_sents, batch_size):

        # batch_src_tokens -> (num_sents x T), batch_src_lengths -> (num_tokens,) contains number of tokens in each sentence without padding
        batch_src_tokens  = src_sents_tokens[start : start + batch_size]
        batch_src_lengths = src_sents_lengths[start : start + batch_size]

        # batch_tgt_tokens_full -> (num_sents x T), batch_tgt_lengths -> (num_tokens,) contains number of tokens in each sentence without padding
        batch_tgt_tokens_full = tgt_sents_tokens[start : start + batch_size]
        batch_tgt_lengths = tgt_sents_lengths[start : start + batch_size]

        # tgt tokens will be      -> <BOS>  I     Love      learning
        # tgt nxt tokens will be  ->   I    Love  learning  EOS
        # This is the setup we need in decoder. So, we are removing EOS in tgt_tokens and BOS marker in tgt_next_tokens
        batch_tgt_tokens = batch_tgt_tokens_full[:, :-1] # Deleting the last column
        batch_tgt_next_tokens = batch_tgt_tokens_full[:, 1:] # Deleting the first column
        batch_tgt_lengths     = batch_tgt_lengths - 1

        total_tokens += jnp.sum(batch_tgt_lengths)

        loss, rnn_params = train_one_batch(rnn_params, batch_src_tokens, batch_src_lengths, batch_tgt_tokens, batch_tgt_lengths, batch_tgt_next_tokens, lr)
        total_loss += loss

    return total_loss / total_tokens, rnn_params

def total_parameters(rnn_params : Dict) -> int:
    embedding_leaves = jax.tree.leaves(rnn_params["embeddings"])
    encoder_leaves = jax.tree.leaves(rnn_params["encoder"])
    decoder_leaves = jax.tree.leaves(rnn_params["decoder"])

    embedding_params = sum(x.size for x in embedding_leaves)
    encoder_params = sum(x.size for x in encoder_leaves)
    decoder_params = sum(x.size for x in decoder_leaves)
    total_params = embedding_params + encoder_params + decoder_params
    return embedding_params, encoder_params, decoder_params, total_params

def train(src_sents_tokens : ArrayLike, src_sents_lengths: ArrayLike, tgt_sents_tokens : ArrayLike, tgt_sents_lengths: ArrayLike, embedding_model: ModelEmbedding, tgt_vocab_size : int, config: TrainConfig):
    hidden_size = config.hidden_size
    embed_size = config.embed_size
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    loss = 0.0

    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"JAX backend: {jax.default_backend()}")
    logger.info(f"local device count: {jax.local_device_count()}")
    logger.info(f"device count: {jax.device_count}")

    rnn_params = init_params(embedding_model.all_src_embeddings(), embedding_model.all_tgt_embeddings(), embed_size, hidden_size, tgt_vocab_size)
    embedding_params, encoder_params, decoder_params, total_params = total_parameters(rnn_params)
    
    logger.info(f"RNN cumulative params : embedding = {embedding_params}, encoder = {encoder_params}, decoder = {decoder_params} and total = {total_params}");

    for epoch in range(1, num_epochs + 1):
        start_time = time.perf_counter()
        loss, rnn_params = train_one_epoch(rnn_params, src_sents_tokens, tgt_sents_tokens, src_sents_lengths, tgt_sents_lengths, batch_size)
        loss.block_until_ready()
        elapsed = time.perf_counter() - start_time
        logger.info(f"Loss at epoch {epoch}/{num_epochs} is {loss}. Time taken to train = {elapsed}")
