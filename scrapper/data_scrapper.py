import io
import os
import random
import time

from PIL import Image
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


class MapDataScrapper:
    def __init__(self, image_size=256, train=1000, val=100, test=20):
        self.image_size = image_size
        self.train = train
        self.val = val
        self.test = test
        self.zoom = 17
        self.bb = [
            37.57078753445211,
            55.77267903541593,
            37.669505960067085,
            55.72968476890415,
        ]  # Moscow center

    def generate_ll(self):
        x1, y1, x2, y2 = self.bb
        return random.uniform(x1, x2), random.uniform(y1, y2)

    def generate_link_yandex(self, ll, sputnik):
        base = "https://yandex.ru/maps/213/moscow/"
        if sputnik:
            base += "sputnik/"

        return f"{base}?ll={ll[0]},{ll[1]}&z={self.zoom}"

    def generate_link_bing(self, ll, sputnik):
        base = f"https://www.bing.com/maps?cp={ll[1]}~{ll[0]}&lvl={self.zoom}"
        if sputnik:
            base += "&style=a"

        return base

    def generate_link_osm(self, ll):
        return f"https://www.openstreetmap.org/#map={self.zoom}/{ll[1]}/{ll[0]}"

    def create_dirs(self, base_path):
        paths = []
        for dir in ("train", "val", "test"):
            path = os.path.join(base_path, dir)
            os.makedirs(path, exist_ok=True)
            paths.append(path)

        return zip(paths, (self.train, self.val, self.test))

    def save_screenshot(self, driver, ll, sputnik, path, i):
        url = self.generate_link_bing(ll, sputnik) if sputnik else self.generate_link_osm(ll)
        print(f"Getting {url}")
        driver.get(url)
        time.sleep(1)
        screenshot_binary = driver.get_screenshot_as_png()
        image = Image.open(io.BytesIO(screenshot_binary))
        width, height = image.size
        left = (width - self.image_size) / 2
        top = (height - self.image_size) / 2
        right = (width + self.image_size) / 2
        bottom = (height + self.image_size) / 2
        image = image.crop((left, top, right, bottom))
        image = image.convert("RGB")

        filename = os.path.join(path, f"{'r' if sputnik else 'l'}_{i}.jpg")
        with open(filename, "w") as file:
            print(filename)
            image.save(file)

    def generate_dataset(self, path):
        options = webdriver.ChromeOptions()
        options.add_argument("--user-data-dir=/tmp/new-chrome-webdriver-profile")
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1440,900")
        driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
        paths = self.create_dirs(path)
        for path, count in paths:
            for i in range(853, count):
                ll = self.generate_ll()
                self.save_screenshot(driver, ll, False, path, i)
                self.save_screenshot(driver, ll, True, path, i)


if __name__ == "__main__":
    m = MapDataScrapper(512, 1100, 0, 0)
    m.generate_dataset(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "map_dataset")
    )
