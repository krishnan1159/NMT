from dataclasses import dataclass
import sys


@dataclass(frozen=True)
class TrainConfig:
    embed_size: int
    hidden_size: int
    num_epochs: int

def is_colab() -> bool:
    return "google.colab" in sys.modules

def get_config() -> TrainConfig:
    if is_colab():
        return TrainConfig(embed_size=128, hidden_size=128, num_epochs=10)

    return TrainConfig(embed_size=1024, hidden_size=1024, num_epochs=1000)
