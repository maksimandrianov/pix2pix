import logging

from net.pix2pix import Direction, Pix2Pix, Pix2PixTrainer

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

WEIGHT_PATH = "bin"
DEBUG_MODE = True
EPOCHS = 10 if DEBUG_MODE else 100

if __name__ == "__main__":
    Pix2PixTrainer(".", "map_dataset", Direction.FORWARD, WEIGHT_PATH, DEBUG_MODE).train(
        EPOCHS, True
    )
    # Pix2Pix(".", "map_dataset", Direction.FORWARD, WEIGHT_PATH, DEBUG_MODE).test(True)
