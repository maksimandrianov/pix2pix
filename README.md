## Pix2Pix Network

### Prepare Weights
```sh
cd bin && ./merge_weight.sh
```

### Usage
```py
from net.pix2pix import Direction, Pix2Pix, Pix2PixTrainer

WEIGHT_PATH = "bin"
EPOCHS = 100

# Learn
p2p_tr = Pix2PixTrainer(".", "map_dataset", Direction.FORWARD, WEIGHT_PATH)
p2p_tr.train(EPOCHS, True)

# Test
p2p = Pix2Pix(".", "map_dataset", Direction.FORWARD, WEIGHT_PATH)
p2p.test(True)
# p2p.transfer_style(your_photo, True)
```