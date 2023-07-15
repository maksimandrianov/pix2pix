import logging

from net.pix2pix import Direction, Pix2Pix, Pix2PixTrainer

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

WEIGHT_PATH = "bin"
DEBUG_MODE = True
EPOCHS = 3 if DEBUG_MODE else 100

if __name__ == "__main__":
    # p2p = Pix2PixTrainer(".", "map_dataset", Direction.FORWARD, WEIGHT_PATH, DEBUG_MODE).train(
    #     EPOCHS, True
    # )
    # p2p.plot()
    Pix2Pix(".", "map_dataset", Direction.FORWARD, WEIGHT_PATH).test(True)
