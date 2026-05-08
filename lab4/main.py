from typing import Any, Callable, List, Dict, Optional
import cv2
import numpy as np
import urllib.request
from pathlib import Path
import re
import matplotlib.pyplot as plt
import sys
import os


def main(*args: Any):
    cur_file_path = Path(__file__).resolve()
    cur_dir_path: Path = cur_file_path.parent
    root_dir_path: Path = cur_dir_path
    imgs_dir_path: Path = root_dir_path / "imgs"

    if not os.path.exists(imgs_dir_path):
        os.mkdir(imgs_dir_path)
    
    urls: List[str] = [
        "https://raw.githubusercontent.com/vision-agh/poc_sw/master/04_Thresholding/coins.png",
        "https://raw.githubusercontent.com/vision-agh/poc_sw/master/04_Thresholding/rice.png",
        "https://raw.githubusercontent.com/vision-agh/poc_sw/master/04_Thresholding/catalogue.png",
        "https://raw.githubusercontent.com/vision-agh/poc_sw/master/04_Thresholding/bart.png",
        "https://raw.githubusercontent.com/vision-agh/poc_sw/master/04_Thresholding/figura1.png",
        "https://raw.githubusercontent.com/vision-agh/poc_sw/master/04_Thresholding/figura2.png",
        "https://raw.githubusercontent.com/vision-agh/poc_sw/master/04_Thresholding/figura3.png",
        "https://raw.githubusercontent.com/vision-agh/poc_sw/master/04_Thresholding/figura4.png",
    ]
    if not os.path.exists(str(imgs_dir_path)):
        raise RuntimeError("imgs_dir_path nie istnieje")
    if not imgs_dir_path.is_dir():
        raise RuntimeError("imgs_dir_path nie jest katalogiem")

    for idx, url in enumerate(urls):
        url_path = Path(url)
        img_path: Path = imgs_dir_path / url_path.name
        
        if not os.path.exists(str(img_path)):
            try:
                urllib.request.urlretrieve(url, str(img_path))
            except Exception as e:
                print(f"otrzymano blad {e}")
    
    imgs: Dict[str, np.ndarray] = {}
    pattern: str = r"\Afigura\d+\.(png|bmp|jpg)\Z"

    for img_path in imgs_dir_path.iterdir():
        img_full_name: str = img_path.name
        found_match: Any = re.search(pattern=pattern, string=img_full_name)
        
        if bool(found_match):
            img_name: str = img_path.stem
            img_extension: str = img_path.suffix

            print(f"stem = {img_name}, suffix = {img_extension}")

            img: np.ndarray = cv2.imread(str(img_path))

            match img.ndim:
                case 2:
                    img = img
                case 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                case _:
                    raise RuntimeError(f"blad przy -> img_name = {img_name}, img.ndim = {img.ndim}")
            imgs[img_name] = img

    n: int = 3*len(imgs)
    max_cols: int = len(imgs)
    cols: int = int(min(n, max_cols))
    rows: int = int(np.ceil(n/cols))
    print(f"cols = {cols}, rows = {rows}")

    _, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = np.atleast_1d(axes).ravel()

    for idx, (key, val) in enumerate(imgs.items()):
        img_name:str = key
        img: np.ndarray = val

        #-----------------------------------
        ax: Any = axes[idx + 0*len(imgs)]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(img_name)
        #-----------------------------------

        #-----------------------------------
        hist: np.ndarray = cv2.calcHist([img], [0], None, [255], [0, 255])
        cum_hist: np.ndarray = np.cumsum(hist)
        norm_cum_hist: np.ndarray = (cum_hist/np.max(cum_hist))*np.max(hist)
        ax: Any = axes[idx + 1*len(imgs)]
        ax.plot(hist)
        ax.plot(norm_cum_hist)
        ax.set_title(f"histogramy dla {img_name}")
        #-----------------------------------

        #-----------------------------------
        k, bin_img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ax: Any = axes[idx + 2*len(imgs)]
        ax.imshow(bin_img_otsu, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"otsu dla {img_name}")
        #-----------------------------------

    for j in range(idx+1, n):
        axes[j].axis("off")

    plt.gray()
    plt.show()


if __name__ == "__main__":
    main()