from dvs_gesture_dataset import DVSGesture
from models import CNN, SNN
from pytorch_lightning.cli import LightningCLI


def cli_main():
    cli = LightningCLI(SNN, DVSGesture)


if __name__ == "__main__":
    cli_main()
