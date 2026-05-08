
from typing import Any, List, Dict, Optional, Callable
from pathlib import Path
import urllib.request
import urllib
import sys
import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt


def main(*args: Any):
    cur_file_path = Path(__file__).resolve()
    root_dir_path: Path = cur_file_path.parent
    imgs_dir_path: Path = root_dir_path / "imgs"

    urls: List[str] = [
        "https://raw.githubusercontent.com/vision-agh/poc_sw/master/03_Histogram/hist1.bmp",
        "https://raw.githubusercontent.com/vision-agh/poc_sw/master/03_Histogram/hist2.bmp",
        "https://raw.githubusercontent.com/vision-agh/poc_sw/master/03_Histogram/hist3.bmp",
        "https://raw.githubusercontent.com/vision-agh/poc_sw/master/03_Histogram/hist4.bmp"
    ]


    if not os.path.exists(imgs_dir_path):
        os.mkdir(imgs_dir_path)
    
    #alternatywnie -> imgs_dir_path.mkdir(exist_ok=True)
        

    if imgs_dir_path.is_dir():
        for url in urls:
            try:
                url_path = Path(url)
                file_name: Path = url_path.name
                urllib.request.urlretrieve(url, str(imgs_dir_path / file_name))
            except Exception as e:
                print(f"otrzyamno blad {e}")

    hist_imgs: Dict[str, np.ndarray] = {}
    hist_imgs_re_pattern: str = r"\Ahist\d+\.(bmp|png|jpg)\Z"

    if imgs_dir_path.is_dir():
        for img_file_path in imgs_dir_path.iterdir():
            found_match: re.Match = re.search(hist_imgs_re_pattern, str(img_file_path.name))
            if found_match:
                #full_name: Any = found_match.group(0)
                name: str = img_file_path.stem
                extension: str = img_file_path.suffix
                #name, extension = tuple(str(full_name).split('.'))

                img: np.ndarray = cv2.imread(str(img_file_path))
                if img is not None:
                    match img.ndim:
                        case 2:
                            img = img
                        case 3:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        case _: 
                            img = img
                    hist_imgs[name] = img

    def apply_clahe(img: np.ndarray):
        if not isinstance(img, np.ndarray):
            raise RuntimeError("img type nie jest np.ndarray")
        if img.ndim != 2:
            raise RuntimeError("img ndim nie jest 2")
        
        clahe: cv2.CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img: np.ndarray = clahe.apply(img)
        return clahe_img

    def apply_eq(img: np.ndarray)->np.ndarray:
        if not isinstance(img, np.ndarray):
            raise RuntimeError("img type nie jest np.ndarray")
        if img.ndim != 2:
            raise RuntimeError("img ndim nie jest 2")   
        eq_img: np.ndarray = cv2.equalizeHist(img) 
        return eq_img
        
    def apply_id(img: np.ndarray)->np.ndarray:
        if not isinstance(img, np.ndarray):
            raise RuntimeError("img type nie jest np.ndarray")
        if img.ndim != 2:
            raise RuntimeError("img ndim nie jest 2")
        return img
    
    funcs_to_compare: List[Callable[[np.ndarray], np.ndarray]] = [
        apply_id,
        apply_eq,
        apply_clahe
    ]

    n: int = len(funcs_to_compare)*len(hist_imgs)
    max_cols: int = len(hist_imgs)
    cols: int = int(min(n, max_cols))
    rows: int = int(np.ceil(n/cols))

    _, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*6))
    axes = np.atleast_1d(axes)
    axes = axes.ravel()

    for img_idx, (key, val) in enumerate(hist_imgs.items()):

        img: np.ndarray = val
        title: str = key

        for func_idx, func in enumerate(funcs_to_compare):
            ax: Any = axes[func_idx*len(hist_imgs)+img_idx]
            try: 
                transformed_img: np.ndarray = func(img)
                ax.imshow(transformed_img, cmap="gray", vmin=0, vmax=255)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"img name: {title}, used func: {func.__name__}")
            except Exception as e:
                ax.axis('off')
    plt.show()



                
                

if __name__ == "__main__":
    main()