import sys
import os

import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from typing import Any, Optional, Callable, List, Tuple, Dict, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

from pathlib import Path
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
from PIL import Image

import re


#-------------------------------------------------------------
def disabled(func: Callable[..., Any]) -> Callable[..., Any]: 
    def wrapper(*args: Any, **kwargs: Any) -> None:
        pass
    return wrapper

def load_imgs_from_url() -> None:
    cur_file_path = Path(__file__).resolve()
    cur_dir_path: Path = cur_file_path.parent
    imgs_dir_path: Path = cur_dir_path / "imgs"

    if not os.path.exists(str(imgs_dir_path)):
        os.mkdir(str(imgs_dir_path))

    repo_url: str = "https://github.com/vision-agh/poc_sw/tree/master/06_Context/"
    raw_base_url: str = "https://raw.githubusercontent.com/vision-agh/poc_sw/master/06_Context/"

    # ogolnie najpierw wchodze na githuba ktory nie ma moich danych
    # aby podkrasc nazwy plikow ktore chce dostac z kontenera

    try:
        with urllib.request.urlopen(repo_url) as response:
            html_content: bytes = response.read()

            soup = BeautifulSoup(html_content, "html.parser")
            
            # 'a' pochodzi od <a> co w html jest sposobem na zakotwiczanie hiperlacza

            links: Set[str] = {a["href"] for a in soup.find_all("a", href=True)}

            
            for link in links:
                if link.endswith((".bmp", ".jpg", ".png")):
                    file_name: str = link.split("/")[-1] # filename to np lena.bmp
                    full_download_url: str = urllib.parse.urljoin(raw_base_url, file_name)

                    with urllib.request.urlopen(full_download_url) as resp:
                        raw_data: bytes = resp.read()
                        img_path: Path = imgs_dir_path/file_name
                        img_path.write_bytes(raw_data)

                    '''                    
                    with urllib.request.urlopen(full_download_url) as resp:
                    # te arraye sa po prostu tab[] bo ich shapey to (n,)
                    img_arr = np.array(bytearray(resp.read()), dtype="uint8")
                    img: np.ndarray = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
                    img_pillow = Image.fromarray(img)
                    img_pillow.save(str(imgs_dir_path/file_name))
                    #ewentualnie cv2.imwrite(img)
                    '''

                    '''                    
                    try:
                        urllib.request.retrieve(full_download_url, str(imgs_dir_path/file_name))
                    except Exception as e:
                        print(f"otryzmano blad {e}")
                    '''

    except Exception as e:
        print(f"otrzymano blad {e}")

def load_imgs_as_dict() -> Dict[str, np.ndarray]:
    cur_file_path = Path(__file__).resolve()
    cur_dir_path: Path = cur_file_path.parent
    imgs_dir_path: Path = cur_dir_path/"imgs"

    if not os.path.exists(str(imgs_dir_path)):
        load_imgs_from_url()

    imgs_as_dict: Dict[str, np.ndarray] = {}
    
    for idx, img_path_str in enumerate(imgs_dir_path.iterdir()):
        img_path = Path(img_path_str)
        img_name: str = img_path.stem
        img_extension: str = img_path.suffix

        if img_extension in {".bmp", ".png", ".jpg"}:
            try: 
                img: np.ndarray = cv2.imread(img_path_str)

                match img.ndim:
                    case 2:
                        pass
                    case 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    case _: 
                        raise RuntimeError(f"img.ndim jest {img.ndim}")

                imgs_as_dict[img_name] = img

            except Exception as e:
                print(f"otrzymano blad {e}")

    return imgs_as_dict

def show_all_imgs(imgs_as_dict: Dict[str, np.ndarray]) -> None:  
    n: int = len(imgs_as_dict)
    plot_cols: int = int(min(np.ceil(np.sqrt(n)), 4))
    plot_rows: int = int(np.ceil(n/plot_cols))

    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(plot_cols*3, plot_rows*3))
    axes = np.atleast_1d(axes).ravel()

    for idx, (key, val) in enumerate(imgs_as_dict.items()):
       ax: Any = axes[idx]
       img: np.ndarray = val
       img_name: np.ndarray = key

       ax.imshow(img, cmap="gray", vmin=0, vmax=255)
       ax.set_xticks([])
       ax.set_yticks([])
       ax.set_title(img_name)

    for j in range(idx+1, n):
        axes[j].axis("off")

    plt.gray()
    plt.show() 
#-------------------------------------------------------------


#-------------------------------------------------------------

@dataclass(frozen=True)
class BaseKernelFactory(ABC):
    class NormType(Enum):
        SUM = 0
        POS_SUM = 1
        ABS_SUM = 2

    size: int 
    kernel_norm_type: NormType = NormType.SUM
    norm_img_flag: bool = False

    kernel: np.ndarray = field(init=False, repr=False)

    @abstractmethod
    def __post_init__(self):
        pass
  
    def _get_normalized_kernel(self, kernel: np.ndarray) -> np.ndarray:
        
        numerator: Optional[float] = None
        match self.kernel_norm_type:
            case self.NormType.SUM:
                numerator= np.sum(kernel)
            case self.NormType.POS_SUM:
                numerator = np.sum(kernel[kernel > 0.0])
            case self.NormType.ABS_SUM:
                numerator = np.sum(np.abs(kernel))

        if numerator is None:
            raise ValueError(f"")
        if np.isclose(numerator, 0.0): 
            raise ValueError(f"")
        
        normalized_kernel: np.ndarray = kernel/numerator
        return normalized_kernel

    def is_img_norm_needed(self) -> bool:
        return self.norm_img_flag
    def __str__(self) -> str:
        return f"{self.__class__.__name__} with size {self.size}"
    def __call__(self) -> np.ndarray:
        return self.kernel

@dataclass(frozen=True)
class CustomKernelFactory(BaseKernelFactory):
    size: int = 3

    def __post_init__(self):
        kernel: np.ndarray = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32) 

        normalized_kernel: np.ndarray = self._get_normalized_kernel(kernel)
        object.__setattr__(self, "kernel", normalized_kernel)

@dataclass(frozen=True)
class IdKernelFactory(BaseKernelFactory):
    size: int = 1

    def __post_init__(self):
        kernel: np.ndarray = np.ones((1, 1), dtype=np.float32)
        normalized_kernel: np.ndarray = self._get_normalized_kernel(kernel)
        object.__setattr__(self, "kernel", normalized_kernel)

@dataclass(frozen=True)
class AvKernelFactory(BaseKernelFactory):
    def __post_init__(self):
        kernel: np.ndarray = np.ones((self.size, self.size), dtype=np.float32)
        normalized_kernel: np.ndarray = self._get_normalized_kernel(kernel)
        object.__setattr__(self, "kernel", normalized_kernel)
    
@dataclass(frozen=True)
class GaussianKernelFactory(BaseKernelFactory):
    size: int = 5
    sigma: float = 0.0

    def __post_init__(self):
        kernel_1d: np.ndarray = cv2.getGaussianKernel(ksize=self.size, sigma=self.sigma)
        kernel_2d: np.ndarray = np.outer(kernel_1d, kernel_1d)
        normalized_kernel: np.ndarray = self._get_normalized_kernel(kernel_2d)
        object.__setattr__(self, "kernel", normalized_kernel)

@dataclass(frozen=True)
class LaplaceKernelFactory(BaseKernelFactory):
    kernel_norm_type = BaseKernelFactory.NormType.ABS_SUM
    norm_img_flag: bool = True
    
    def __post_init__(self): 
        kernel: np.ndarray = np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=np.float32)

        normalized_kernel: np.ndarray = self._get_normalized_kernel(kernel)
        object.__setattr__(self, "kernel", normalized_kernel)

def zad1() -> None:
    imgs_as_dict: Dict[str, np.ndarray] = load_imgs_as_dict()
    img_name: str = "plansza"
    img: np.ndarray = imgs_as_dict[img_name]

    kernel_factories: List[BaseKernelFactory] = [
        IdKernelFactory(),
        CustomKernelFactory(),
        GaussianKernelFactory()  
    ]

    for kernel_size in (29, ):
        kernel_factories.append(AvKernelFactory(kernel_size))  
    
    n: int = len(kernel_factories)
    max_plot_cols: int = 4
    plot_cols: int = int(min(np.ceil(np.sqrt(n)), max_plot_cols))
    plot_rows: int = int(np.ceil(n/plot_cols))

    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(plot_cols*3, plot_rows*3))
    axes = np.atleast_1d(axes).ravel()

    for idx, kernel_factory in enumerate(kernel_factories):
        kernel: np.ndarray = kernel_factory()
        img_filtered: np.ndarray = cv2.filter2D(src=img, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT_101)

        ax: Any = axes[idx]
        ax.imshow(img_filtered, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{img_name} -> {str(kernel_factory)}")
    
    plt.gray()
    plt.show()
    
#-------------------------------------------------------------


def main(args=None):
    load_imgs_from_url()
    zad1()

if __name__ == "__main__":
    main()
