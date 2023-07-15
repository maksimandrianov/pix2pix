## Pix2Pix Network

### Learning process
Images from the validation dataset at different epochs
![Pix2Pix Training](https://github.com/maksimandrianov/pix2pix/blob/main/bin/1_progress.gif?raw=true)
![Pix2Pix Training](https://github.com/maksimandrianov/pix2pix/blob/main/bin/2_progress.gif?raw=true)

![Pix2Pix Metrics1](https://github.com/maksimandrianov/pix2pix/blob/main/bin/metrics1.png?raw=true)
![Pix2Pix Metrics2](https://github.com/maksimandrianov/pix2pix/blob/main/bin/metrics2.png?raw=true)

### Dataset

It uses its own dataset of satellite images from Bing and map schemes from OSM.
See [scrapper/data_scrapper.py](https://github.com/maksimandrianov/pix2pix/blob/main/scrapper/data_scrapper.py)

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

# Or Test with weights loaded
p2p = Pix2Pix(".", "map_dataset", Direction.FORWARD, WEIGHT_PATH)
p2p.test(True)
# p2p.transfer_style(your_photo, True)
```