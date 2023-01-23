from dvs_gesture_dataset import DVSGesture
from models import CNN, SNN
from pytorch_lightning.cli import LightningCLI


class LinkedLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.batch_size", "model.batch_size")


def cli_main():
    cli = LinkedLightningCLI(CNN, DVSGesture)


if __name__ == "__main__":
    cli_main()
