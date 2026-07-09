from dataclasses import dataclass, replace
import sys


@dataclass(frozen=True)
class TrainConfig:
    embed_size: int
    hidden_size: int
    num_epochs: int
    batch_size: int


def is_colab() -> bool:
    return "google.colab" in sys.modules


def get_config(**overrides) -> TrainConfig:
    if is_colab():
        config = TrainConfig(
            embed_size=256,
            hidden_size=256,
            num_epochs=1000,
            batch_size=128,
        )
    else:
        config = TrainConfig(
            embed_size=64,
            hidden_size=64,
            num_epochs=100,
            batch_size=32,
        )

    return replace(config, **overrides)