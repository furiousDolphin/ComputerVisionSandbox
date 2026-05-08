
import sys
import os

import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from typing import Any, Optional, Callable, List, Tuple, Dict, Set
from dataclasses import dataclass

from pathlib import Path
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
from PIL import Image

import re



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

    repo_url: str = "https://github.com/vision-agh/poc_sw/tree/master/08_Fourier/"
    raw_base_url: str = "https://raw.githubusercontent.com/vision-agh/poc_sw/master/08_Fourier/"

    # ogolnie najpierw wchodze na githuba ktory nie ma moich danych
    # aby podkrasc nazwy plikow ktore chce dostac z kontenera

    try:
        with urllib.request.urlopen(repo_url) as response:
            html_content: bytes = response.read()

            soup = BeautifulSoup(html_content, "html.parser")
            
            # 'a' pochodzi od <a> co w html jest sposobem na zakotwiczanie hiperlacza

            links: Set[str] = {a["href"] for a in soup.find_all("a", href=True)}

            for link in links:
                if link.endswith(".bmp"):
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


#-------------------------------------------------------------------------
def zad1_computing_2dFImg_as_two_fft(img: np.ndarray) -> np.ndarray:
    img_f: np.ndarray = np.float32(img)
    FRows: np.ndarray = np.fft.fft(img_f, axis=0)
    FRowsCols: np.ndarray = np.fft.fft(FRows, axis=1)
    FReal: np.ndarray = FRowsCols.real 
    FImag: np.ndarray = FRowsCols.imag
    return np.stack((FReal, FImag), axis=-1) #alternatywnie cv2.merge([FReal, FImag])

def zad1_get_FImg_mag_phase(img: np.ndarray, fft_func: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
   
    FImg: Optional[np.ndarray] = None

    if fft_func is None:
        FImg = cv2.dft(src=np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    else:
        FImg = fft_func(img)

    FImg_shifted: np.ndarray = np.fft.fftshift(FImg, [0, 1])
    FImg_real: np.ndarray = FImg_shifted[:, :, 0] 
    FImg_imag: np.ndarray = FImg_shifted[:, :, 1]
    mag, phase = cv2.cartToPolar(FImg_real, FImg_imag)
    mag_log: np.ndarray = 20.0*np.log10(mag+1)

    return mag_log, phase

def zad1_methods_comparision(img: np.ndarray) -> np.ndarray:
    dft_cv2: np.ndarray = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_own: np.ndarray = zad1_computing_2dFImg_as_two_fft(img)

    diff: np.ndarray = cv2.absdiff(np.float32(dft_cv2), np.float32(dft_own))

    return diff

def zad1_part1() -> None:
    imgs_as_dict: Dict[str, np.ndarray] = load_imgs_as_dict()
    
    img_names: Set[str] = {
        "dwieFale",
        "kolo", 
        "kwadrat", 
        "kwadrat45", 
        "trojkat"
    }

    plot_rows: int = len(img_names)
    plot_cols: int = 3

    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(plot_cols*3, plot_rows*3))
    axes = np.atleast_1d(axes).ravel()

    for idx, img_name in enumerate(img_names):
        if img_name in imgs_as_dict.keys():

            img: np.ndarray = imgs_as_dict[img_name]
            mag_log, phase = zad1_get_FImg_mag_phase(img)

            ax: Any = axes[plot_cols*idx + 0]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{img_name}")

            ax: Any = axes[plot_cols*idx + 1]
            ax.imshow(mag_log, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"mag_log")

            ax: Any = axes[plot_cols*idx + 2]
            ax.imshow(phase, cmap="gray", vmin=0, vmax=2*np.pi)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"phase")

    plt.gray()
    plt.show()

def zad1_part2() -> None:
    imgs_as_dict: Dict[str, np.ndarray] = load_imgs_as_dict()
    
    img_names: Set[str] = {
        "kolo", 
        "kwadrat", 
        "kwadrat45", 
        "trojkat"
    }

    plot_rows: int = len(img_names)
    plot_cols: int = 2

    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(plot_cols*3, plot_rows*3))
    axes = np.atleast_1d(axes).ravel()

    for idx, img_name in enumerate(img_names):
        if img_name in imgs_as_dict.keys():

            img: np.ndarray = imgs_as_dict[img_name]
            FDiff = zad1_methods_comparision(img)

            ax: Any = axes[plot_cols*idx + 0]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{img_name}")

            ax: Any = axes[plot_cols*idx + 1]
            ax.imshow(FDiff[:, :, 0], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"FDiff")

    plt.gray()
    plt.show()

@disabled
def zad1() -> None:
    zad1_part1()
    zad1_part2()
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
def zad2_part1() -> None:
    pattern: str = r"\Akwadrat.*\Z"

    imgs_as_dict: Dict[str, np.ndarray] = load_imgs_as_dict()
    img_names_lst: List[str] = []

    for idx, img_name in enumerate(imgs_as_dict.keys()):
        found_match: re.Match = re.search(pattern, img_name)
        if bool(found_match):
            img_names_lst.append(img_name)

    plot_rows: int = len(img_names_lst)
    plot_cols: int = 3

    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(plot_cols*3, plot_rows*3))
    axes = np.atleast_1d(axes).ravel()

    for idx, img_name in enumerate(img_names_lst):
       
        img: np.ndarray = imgs_as_dict[img_name]
        mag_log, phase = zad1_get_FImg_mag_phase(img)

        ax: Any = axes[plot_cols*idx + 0]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{img_name}")

        ax: Any = axes[plot_cols*idx + 1]
        ax.imshow(mag_log, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"mag_log")

        ax: Any = axes[plot_cols*idx + 2]
        ax.imshow(phase, cmap="gray", vmin=0, vmax=2*np.pi)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"phase")

    plt.gray()
    plt.show()

@disabled
def zad2() -> None:
    zad2_part1()
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
def zad3_dft(img: np.ndarray) -> np.ndarray:
    FImg: np.ndarray = cv2.dft(src=np.float32(img), flags=(cv2.DFT_COMPLEX_OUTPUT))
    FImg_shifted: np.ndarray = np.fft.fftshift(FImg, [0, 1])
    return FImg_shifted

def zad3_idft(FImg_shifted: np.ndarray) -> np.ndarray:
    FImg: np.ndarray = np.fft.ifftshift(FImg_shifted, [0, 1])
    img_f: np.ndarray = cv2.idft(src=FImg, flags=(cv2.DFT_SCALE|cv2.DFT_COMPLEX_OUTPUT))
    img_f = np.round(cv2.magnitude(img_f[:, :, 0], img_f[:, :, 1]))
    return np.uint8(img_f)

def zad3_part1() -> None:
    imgs_as_dict: Dict[str, np.ndarray] = load_imgs_as_dict()
    img_to_test: np.ndarray = imgs_as_dict["kolo"]
    img: np.ndarray = cv2.imread("kolo.bmp", cv2.IMREAD_GRAYSCALE)
    
    FImg: np.ndarray = zad3_dft(img_to_test)
    IFImg: np.ndarray = zad3_idft(FImg)

    rows: int = 1
    cols: int = 2
    _, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = np.atleast_1d(axes).ravel()

    ax: Any = axes[0]
    ax.imshow(img_to_test, cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"img_to_test")

    ax: Any = axes[1]
    ax.imshow(np.abs(IFImg - img_to_test), cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"abs(IFImg - img_to_test)")

    plt.gray()
    plt.show()

@disabled
def zad3() -> None:
    zad3_part1()
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
def zad4_get_FImg_mag_phase(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    FImg: np.ndarray = cv2.dft(src=np.float32(img), flags=(cv2.DFT_COMPLEX_OUTPUT))
    FImg_real: np.ndarray = FImg[:, :, 0]
    FImg_imag: np.ndarray = FImg[:, :, 1]
    mag, phase = cv2.cartToPolar(FImg_real, FImg_imag)
    mag_log: np.ndarray = 20*np.log10(mag+1)
    return (mag_log, phase)

def zad4_create_base_filter(filter_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  
    FSpaceRows: np.ndarray = 2.0*np.fft.fftshift(np.fft.fftfreq(filter_size[0]), axes=None)
    FSpaceRowsM: np.ndarray = np.outer(FSpaceRows, np.ones([1, filter_size[1]]))
    FSpaceCols: np.ndarray = 2.0*np.fft.fftshift(np.fft.fftfreq(filter_size[1]), axes=None)
    FSpaceColsM: np.ndarray = np.outer(np.ones([1, filter_size[0]]), FSpaceCols)
    
    FreqR: np.ndarray = np.sqrt(np.square(FSpaceRowsM)+np.square(FSpaceColsM))
    return FreqR, FSpaceRowsM, FSpaceColsM

def zad4_get_FImg_shifted(img: np.ndarray) -> np.ndarray:
    FImg: np.ndarray = cv2.dft(src=np.float32(img), flags=(cv2.DFT_COMPLEX_OUTPUT))
    FImg_shifted: np.ndarray = np.fft.fftshift(FImg, axes=[0, 1])
    return FImg_shifted

def zad4_get_img_f_from_FImg_shifted(FImg_shifted: np.ndarray) -> np.ndarray:
    FImg: np.ndarray = np.fft.ifftshift(FImg_shifted, axes=[0, 1])
    img_f: np.ndarray = cv2.idft(src=FImg, flags=(cv2.DFT_COMPLEX_OUTPUT|cv2.DFT_SCALE))
    return img_f

def zad4_apply_filter(img: np.ndarray, FilterF: np.ndarray) -> np.ndarray:
    FImg_shifted: np.ndarray = zad4_get_FImg_shifted(img)
    FilterF3: np.ndarray = np.repeat(FilterF[:, :, np.newaxis], 2, axis=2)
    FImg_filtered: np.ndarray = FImg_shifted*FilterF3
    img_filtered3: np.ndarray = zad4_get_img_f_from_FImg_shifted(FImg_filtered)
    img_filtered: np.ndarray = cv2.magnitude(img_filtered3[:, :, 0], img_filtered3[:, :, 1])
    return np.uint8(img_filtered)

def zad4_part1():
    imgs_as_dict: Dict[str, np.ndarray] = load_imgs_as_dict()
    img: np.ndarray = imgs_as_dict["lena"]

    FreqR, FSpaceRowsM, FSpaceColsM = zad4_create_base_filter(img.shape)

    FilterF_lambdas_dict: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "dolno": lambda FreqR: (FreqR < 0.3),
        "pasmowo": lambda FreqR: (FreqR > 0.1)*(FreqR < 0.5),
        "gorno": lambda Freq: (Freq > 0.3)
    }

    plot_rows: int = len(FilterF_lambdas_dict)
    plot_cols: int = 3

    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(plot_cols*3, plot_rows*3))
    axes = np.atleast_1d(axes).ravel()

    for idx, (FilterF_name, FilterF_lambda) in enumerate(FilterF_lambdas_dict.items()):
       
        FilterF: np.ndarray = FilterF_lambda(FreqR)
        img_filtered: np.ndarray = zad4_apply_filter(img, FilterF)

        ax: Any = axes[plot_cols*idx + 0]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"lena")
        ax.axis("off")

        ax: Any = axes[plot_cols*idx + 1]
        ax.imshow(img_filtered, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"lena przefiltrowana {FilterF_name}")
        ax.axis("off")

        axes[plot_cols*idx + 2].remove()  

        ax3d = fig.add_subplot(plot_rows, plot_cols, plot_cols*idx + 3, projection="3d")
        ax3d.plot_surface(FSpaceRowsM, FSpaceColsM, FilterF, cmap="gray")
        ax3d.set_title(f"{FilterF_name}")

    plt.gray()
    plt.show()   

@disabled
def zad4() -> None:
    zad4_part1()
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------

def zad5_get_FImg(img: np.ndarray) -> np.ndarray:
    FImg: np.ndarray = cv2.dft(src=np.float32(img), flags=(cv2.DFT_COMPLEX_OUTPUT))
    return FImg

def zad5_get_img_f_from_IFmg(FImg) -> np.ndarray:
    img_f: np.ndarray = cv2.idft(src=FImg, flags=(cv2.DFT_COMPLEX_OUTPUT|cv2.DFT_SCALE))
    return img_f

def zad5_make_complex(FImg: np.ndarray) -> np.ndarray:
    FImg_real: np.ndarray = FImg[:, :, 0]
    FImg_imag: np.ndarray = FImg[:, :, 1]
    FImg_complex: np.ndarray = FImg_real + FImg_imag*1j
    return FImg_complex

def zad5_part1() -> None:
    imgs_as_dict: Dict[str, np.ndarray] = load_imgs_as_dict()

    img_literki: np.ndarray = imgs_as_dict["literki"]
    img_wzorA_rot: np.ndarray = np.rot90(imgs_as_dict["wzorA"], 2)

    literki_rows, literki_cols = img_literki.shape
    wzorA_rows, wzorA_cols = img_wzorA_rot.shape
    delta_rows: int = literki_rows - wzorA_rows
    delta_cols: int = literki_cols - wzorA_cols

    img_wzorA_rot_padded: np.ndarray = cv2.copyMakeBorder(
        img_wzorA_rot,
        0, delta_rows,
        0, delta_cols,
        cv2.BORDER_CONSTANT,
        value=0
    )

    FImg_wzorA_rot_padded: np.ndarray = zad5_get_FImg(img_wzorA_rot_padded)
    FImg_literki: np.ndarray = zad5_get_FImg(img_literki)

    FImg_literki_complex: np.ndarray = zad5_make_complex(FImg_literki)
    FImg_wzorA_rot_padded_complex: np.ndarray = zad5_make_complex(FImg_wzorA_rot_padded)
    
    cor_complex_res: np.ndarray = FImg_literki_complex*FImg_wzorA_rot_padded_complex

    FCompMat: np.ndarray = np.stack((cor_complex_res.real, cor_complex_res.imag), axis=-1)
    #FCompMat: np.ndarray = cv2.merge([cor_complex_res.real, cor_complex_res.imag])

    CompMat: np.ndarray = zad5_get_img_f_from_IFmg(FCompMat)
    CompMat_mag: np.ndarray = cv2.magnitude(CompMat[:, :, 0], CompMat[:, :, 1])
    CompMat_mag_norm = np.zeros_like(CompMat_mag)
    cv2.normalize(CompMat_mag, CompMat_mag_norm, 0, 255, cv2.NORM_MINMAX)
    CompMat_uint8 = CompMat_mag_norm.astype(np.uint8)
    local_extremes: np.ndarray = cv2.morphologyEx(CompMat_uint8, cv2.MORPH_TOPHAT, np.ones((3, 3), np.uint8))

    imgs_to_show: Dict[str, np.ndarray] = {}
    imgs_to_show["wzorA_rot"] = img_wzorA_rot
    imgs_to_show["wzorA_rot_padded"] = img_wzorA_rot_padded
    imgs_to_show["literki"] = img_literki
    imgs_to_show["local_extremes"] = local_extremes


    n: int = len(imgs_to_show)
    plot_cols: int = int(min(np.ceil(np.sqrt(n)), 4))
    plot_rows: int = int(np.ceil(n/plot_cols))

    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(plot_cols*3, plot_rows*3))
    axes = np.atleast_1d(axes).ravel()

    for idx, (img_name, img) in enumerate(imgs_to_show.items()):

        ax: Any = axes[idx]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{img_name}")

    plt.gray()
    plt.show()

#@disabled
def zad5() -> None:
    zad5_part1()
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
def zad6_make_complex(FImg: np.ndarray) -> np.ndarray:
    FImg_real: np.ndarray = FImg[:, :, 0]
    FImg_imag: np.ndarray = FImg[:, :, 1]
    FImg_complex: np.ndarray = FImg_real + FImg_imag*1j
    return FImg_complex

def zad6_part1() -> None:
    filter_size: int = 21
    hanning_window_1d: np.ndarray = np.hanning(filter_size)
    hanning_window_2d: np.ndarray = np.outer(hanning_window_1d, hanning_window_1d)
    hanning_window_2d = np.stack((hanning_window_2d, hanning_window_2d), axis=-1)

    FSpaceRows: np.ndarray = 2.0*np.fft.fftshift(np.fft.fftfreq(filter_size))
    FSpaceRowsM: np.ndarray = np.outer(FSpaceRows, np.ones([1, filter_size]))
    FSpaceCols: np.ndarray = 2.0*np.fft.fftshift(np.fft.fftfreq(filter_size))
    FSpaceColsM: np.ndarray = np.outer(FSpaceCols, np.ones([1, filter_size]))
   
    FreqR: np.ndarray = np.sqrt(np.square(FSpaceRowsM)+np.square(FSpaceColsM))
    FilterF: np.ndarray = (FreqR > 0.5)

    FilterFRot: np.ndarray = np.rot90(np.fft.fftshift(np.rot90(FilterF, 2)), 2)
    FilterFRot3: np.ndarray = np.stack((FilterFRot, np.zeros(FilterFRot.shape)), axis=-1)
    Filter: np.ndarray = cv2.idft(src=np.float32(FilterFRot3), flags=(cv2.DFT_SCALE|cv2.DFT_COMPLEX_OUTPUT))
    Filter_centered = np.fft.fftshift(Filter, [0, 1])
    Filter_win: np.ndarray = Filter_centered*hanning_window_2d

    imgs_as_dict: Dict[str, np.ndarray] = load_imgs_as_dict()
    img: np.ndarray = imgs_as_dict["lena"]
    FImg: np.ndarray = cv2.dft(src=np.float32(img), flags=(cv2.DFT_COMPLEX_OUTPUT))

    img_rows, img_cols = img.shape
    delta_rows: int = img_rows - filter_size
    delta_cols: int = img_cols - filter_size

    Filter_win_pad: np.ndarray = cv2.copyMakeBorder(
        Filter_win,
        int(np.floor(delta_rows/2)), int(np.ceil(delta_rows/2)),
        int(np.floor(delta_cols/2)), int(np.ceil(delta_cols/2)),
        cv2.BORDER_CONSTANT,
        value=0
    )

    Filter_win_pad_shifted: np.ndarray = np.fft.ifftshift(Filter_win_pad, [0, 1])
    FilterF_win_pad_shifted: np.ndarray = cv2.dft(src=Filter_win_pad_shifted, flags=(cv2.DFT_COMPLEX_OUTPUT))

    FilterF_win_pad_shifted_complex: np.ndarray = zad6_make_complex(FilterF_win_pad_shifted)
    FImg_complex: np.ndarray = zad6_make_complex(FImg)

    FImg_filtered_complex: np.ndarray = FilterF_win_pad_shifted_complex*FImg_complex
    FImg_filtered: np.ndarray = np.stack((FImg_filtered_complex.real, FImg_filtered_complex.imag), axis=-1)
    img_filtered: np.ndarray = cv2.idft(src=FImg_filtered, flags=(cv2.DFT_SCALE|cv2.DFT_COMPLEX_OUTPUT))
    img_filtered = np.float32(img_filtered)
    img_filtered_mag: np.ndarray = cv2.magnitude(img_filtered[:, :, 0], img_filtered[:, :, 1])
    img_filtered_mag = np.uint8(img_filtered_mag)

    plt.figure()
    plt.imshow(img_filtered_mag, cmap="gray", vmin=0, vmax=255)
    plt.xticks([])
    plt.yticks([])
    plt.gray()
    plt.show()

@disabled
def zad6() -> None:
    zad6_part1()
#-------------------------------------------------------------------------

def main(args: Any = None):
    # imgs_as_dict: Dict[str, np.ndarray] = load_imgs_as_dict()
    # show_all_imgs(imgs_as_dict)
    zad1()
    zad2()
    zad3()
    zad4()
    zad5()
    zad6()

if __name__ == "__main__":
    main()