from pathlib import Path
from typing import List, Tuple
import sentencepiece as spm
import numpy as np
import numpy.typing as npt
import jax
from jax.typing import ArrayLike
import jax.numpy as jnp

TOKENIZER_ARTIFACTS_DIR = Path("artifacts/tokenizer")


class Tokenizer(object):

    """
    Each line is represented as one single string along with whitespaces. So we have List[List[str]]
    """
    def __init__(self, sentences : List[str], fname : str, model_prefix : str, vocab_size : int = 8000):
        self.sentences = sentences
        self.fname = Tokenizer._artifact_path(fname)
        self.model_prefix = Tokenizer._artifact_path(model_prefix)
        ## Write to file
        Tokenizer._write_to_file(sentences, self.fname)

        spm.SentencePieceTrainer.Train(input=str(self.fname), model_prefix=str(self.model_prefix), vocab_size=int(vocab_size), hard_vocab_limit=False, pad_id = 0, unk_id=1, bos_id=2, eos_id=3, minloglevel=2)

        self.model = spm.SentencePieceProcessor(model_file=f"{self.model_prefix}.model")
        self._vocab_size = self.model.GetPieceSize()
    
    def to_numpy(self, sentences : List[str], add_bos : bool = False, add_eos : bool = False) -> Tuple[npt.NDArray, npt.NDArray]:
        if len(sentences) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        all_tokens = []
        for sentence in sentences:
            ids = []
            if add_bos:
                ids.append(self.model.bos_id())

            ids.extend(self.model.EncodeAsIds(sentence))

            if add_eos:
                ids.append(self.model.eos_id())

            all_tokens.append(ids)
        
        lengths = [len(sentence_token) for sentence_token in all_tokens]
        max_len = max(lengths)

        for sentence_tokens in all_tokens:
            sentence_tokens.extend([self.model.pad_id()] * (max_len - len(sentence_tokens)))
        
        return np.array(all_tokens, dtype=np.int64), np.array(lengths, dtype=np.int64)
    
    def get_vocab_size(self) -> int:
        return self.model.GetPieceSize()

    @staticmethod
    def _artifact_path(name : str):
        path = Path(name)
        if path.parent == Path("."):
            return TOKENIZER_ARTIFACTS_DIR / path
        return path

    
    @staticmethod
    def _write_to_file(sentences : List[str], file_name):
        file_name.parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, "w", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(sentence + "\n")
    

if __name__ == "__main__":
    corpus = [
        "It is a beautiful world",
        "I love NLP",
        "FiFa worldcup is boring",
        "Lost interest in cricket",
        "Focus to succeed"
    ]

    vocab = Tokenizer(corpus, "corpus", "corpus")

    test = [
        "love to succeed",
        "world is boring"
    ]

    print(f"Encoded version of {test} is {vocab.to_numpy(test)}")
