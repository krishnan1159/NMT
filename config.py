from dataclasses import dataclass
import sys


@dataclass(frozen=True)
class TrainConfig:
    embed_size: int
    hidden_size: int
    num_epochs: int
    batch_size: int

def is_colab() -> bool:
    return "google.colab" in sys.modules

def get_config() -> TrainConfig:
    if is_colab():
        return TrainConfig(embed_size=256, hidden_size=256, num_epochs=1000, batch_size=128)

    return TrainConfig(embed_size=256, hidden_size=256, num_epochs=10, batch_size=64)
