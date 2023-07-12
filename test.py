import logging

from net.pix2pix import Direction, Pix2Pix, Pix2PixTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

WEIGHT_PATH = "bin"
TEST_MODE = True

if __name__ == "__main__":
    epochs = 10 if TEST_MODE else 100
    Pix2PixTrainer(".", "map_dataset", Direction.FORWARD, WEIGHT_PATH, epochs, TEST_MODE).train(
        True
    )
    # Pix2Pix(
    #     ".", "map_dataset", Direction.FORWARD, WEIGHT_PATH, TEST_MODE
    # ).test(True)
