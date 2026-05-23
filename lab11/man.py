from __future__ import annotations

import numpy as np
import scipy as sp
import cv2
import matplotlib.pyplot as plt
import skimage.transform
import urllib.request
import urllib.parse
import re
import os

from pathlib import Path
from typing import Optional, Any, Callable, List, Set, Tuple, Dict
from dataclasses import dataclass
from bs4 import BeautifulSoup
from PIL import Image

def get_root_dir_path() -> Optional[Path]:
    cur_file_path = Path(__file__).resolve()

    for dir_path in cur_file_path.parents:
        if os.path.exists(str(dir_path/"imgs")):
            return dir_path
    return None

def load_imgs_as_dict() -> Dict[str, np.ndarray]:

    root_dir_path: Optional[Path] = get_root_dir_path()
    if root_dir_path is None:
        raise RuntimeError("nie znaleziono roota")
    
    imgs_dir_path: Path = root_dir_path/"imgs"
    if not imgs_dir_path.exists() or not imgs_dir_path.is_dir():
        raise RuntimeError(f"nie znaleziono katalogu {imgs_dir_path}")
    
    pattern: str = r"\A.*(.png|.jpg|.bmp)\Z"
    imgs_as_dict: Dict[str, np.ndarray] = {}
    for idx, img_path in enumerate(imgs_dir_path.iterdir()):
        img_full_name: str = img_path.name
        found_match: re.Match = re.search(pattern, img_full_name)
        if bool(found_match):
            img_name: str = img_path.stem
            img: np.ndarray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            imgs_as_dict[img_name] = img
    return imgs_as_dict

def show_imgs(imgs_as_dict: Dict[str, np.ndarray]) -> None:
    n: int = len(imgs_as_dict)
    cols_max: int = 4
    cols: int = int(min(np.ceil(np.sqrt(n)), cols_max))
    rows: int = int(np.ceil(n/cols))

    _, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = np.atleast_1d(axes).ravel()
    
    for idx, (img_name, img_cur) in enumerate(imgs_as_dict.items()):

        ax: Any = axes[idx]
        ax.imshow(img_cur, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{img_name}")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.gray()
    plt.show()

def download_imgs_from_url():
    cur_file_path = Path(__file__).resolve()
    root_dir_path: Optional[Path] = None

    for dir_path in cur_file_path.parents:
        if os.path.exists(str(dir_path/"imgs")):
            root_dir_path = dir_path

    imgs_dir_path: Path = root_dir_path/"imgs"
    if not imgs_dir_path.exists():
        os.mkdir(str(imgs_dir_path))
    
    repo_url: str = "https://github.com/vision-agh/poc_sw/tree/master/11_Hough/"
    raw_base_url: str = "https://raw.githubusercontent.com/vision-agh/poc_sw/master/11_Hough/"

    try:
        with urllib.request.urlopen(repo_url) as response:
            html_content: bytes = response.read()
            soup = BeautifulSoup(html_content, "html.parser")
            links: Set[str] = {a["href"] for a in soup.find_all("a", href=True)}
            
            pattern: str = r"\A.*(.bmp|.jpg|.png)\Z"
            for link in links:
                file_path = Path(str(link))
                file_full_name: str = file_path.name
                found_match: re.Match = re.search(pattern, file_full_name)

                if bool(found_match):
                    full_download_url: str = urllib.parse.urljoin(raw_base_url, file_full_name)

                    with urllib.request.urlopen(full_download_url) as resp:
                        raw_data: bytes = resp.read()
                        img_path: Path = imgs_dir_path/file_full_name
                        if not img_path.exists():
                            img_path.write_bytes(raw_data)

    except Exception as e:
        print(f"otrzymano blad {e}")


def func():
    cur_file_path = Path(__file__).resolve()
    root_dir_path: Path = get_root_dir_path()
    imgs_dir_path: Path = root_dir_path/"imgs"

    img_full_name: str = "patelnia1.jpg"
    img_color: np.ndarray = cv2.imread(str(imgs_dir_path/img_full_name), cv2.IMREAD_COLOR_RGB)
    img_gray: np.ndarray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    img_grid: np.ndarray = img_gray.copy()

    tilesize: int = 200
    height, width = img_gray.shape
    height_grid, width_grid = height//tilesize, width//tilesize

    frame_cur: np.ndarray = np.zeros((height, width), dtype=np.uint8)
    frame_prev: np.ndarray = np.zeros_like(frame_cur, dtype=np.uint8)

    img_means: np.ndarray = np.zeros((height_grid, width_grid), dtype=np.float32)
    img_stds: np.ndarray = np.zeros_like(img_means, dtype=np.float32)
    img_bin: np.ndarray = np.zeros_like(img_means, dtype=np.float32)

    y0: int = int((height-height_grid*tilesize)/2)
    x0: int = int((width-width_grid*tilesize)/2)

    flow: np.ndarray = cv2.calcOpticalFlowFarneback(
        frame_prev, frame_cur, flow=None,
        pyr_scale=0.5, levels=3, winsize=15, 
        iterations=3, poly_n=5, poly_sigma=1.2, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    
    mag, angle = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    mag_grid: np.ndarray = np.zeros((height_grid, width_grid), dtype=np.uint8)

    for i in range(height_grid):
        for j in range(width_grid):
            y, x = i*tilesize+y0, j*tilesize + x0

            frame_cur_block: np.ndarray = img_gray[y:(y+tilesize), x:(x+tilesize)].astype(np.float32)
            img_means[i, j] = np.mean(frame_cur_block)
            img_stds[i, j] = np.std(frame_cur_block)

            mag_block: np.ndarray = mag[y:(y+tilesize), x:(x+tilesize)].astype(np.float32)
            mag_grid[i, j] = np.mean(mag_block)

            cv2.rectangle(img_grid, (x, y), (x+tilesize, y+tilesize), (255), 1)

    img_variances: np.ndarray = np.square(img_stds)
    img_bin: np.ndarray = (img_variances<40)

    mag_grid_norm: Optional[np.ndarray] = None
    mag_grid_bin: Optional[np.ndarray] = None
    mag_min: float = np.min(mag_grid)
    mag_max: float = np.max(mag_grid)

    if (mag_max-mag_min) < 1e-3:
        mag_grid_norm = np.zeros_like(mag_grid, dtype=np.uint8)
        mag_grid_bin = np.zeros_like(mag_grid, dtype=np.uint8)
    else:
        mag_grid_norm = (255.0*(mag_grid-mag_min)/(mag_max-mag_min)).astype(np.uint8)
        res_otsu: Tuple[int, np.ndarray] = cv2.threshold(mag_grid_norm, thresh=0, maxval=255, type=cv2.THRESH_OTSU|cv2.THRESH_BINARY)
        mag_grid_bin = res_otsu[1]


    imgs_to_show: Dict[str, np.ndarray] = {
        "img_gray": img_gray,
        "img_grid": img_grid,
        "img_means": img_means.astype(np.uint8),
        "img_stds": img_stds.astype(np.uint8),
        "img_bin": img_bin
    }

    show_imgs(imgs_to_show)

def main(args=None):
    func()



if __name__ == "__main__":
    main()