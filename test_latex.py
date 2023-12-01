from latex import build_pdf
from pdf2image import convert_from_bytes
import io
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from toolbox.printing import debug
import numpy as np
import cv2
import clip
import torch
from dataclasses import dataclass


def latex_to_pdf(latex_code: str) -> io.BytesIO:
    # Compiling LaTeX code to PDF
    pdf = build_pdf(latex_code)
    return io.BytesIO(pdf.data)  # Convert PDF to a byte stream


def pdf_to_image(pdf_stream: io.BytesIO) -> Image:
    # Convert the first page of the PDF stream to an image
    images = convert_from_bytes(pdf_stream.read(), first_page=1, last_page=1)
    if images:
        return images[0]
    else:
        raise Exception("PDF to Image conversion failed")


def latex_to_image(latex_code: str) -> Image:
    try:
        pdf_stream = latex_to_pdf(latex_code)
        image = pdf_to_image(pdf_stream)
        return image
    except Exception as e:
        print(f"An error occurred: {e}")


device, model, preprocess = None, None, None


def compute_clip(image: Image) -> torch.Tensor:
    global device, model, preprocess
    if device is None:
        print("Loading CLIP... ", end="")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("Done!")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features


def compute_ssim(image1: Image, image2: Image) -> float:
    image1 = np.array(image1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = np.array(image2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return ssim(image1, image2)


@dataclass
class LatexResults:
    ssim: float
    score_ssim: float
    clip_distance: float
    score_clip: float


def evaluate_latex(source_code: str, generated_code: str) -> LatexResults:
    """Evaluate the similarity between two latex codes

    Args:
        source_code (str): Source latex code
        generated_code (str): Generated latex code

    Returns:
        LatexResults: Results of the evaluation
    """
    gt_image: Image = latex_to_image(source_code)
    pred_image: Image = latex_to_image(generated_code)
    image_empty = Image.new("RGB", gt_image.size, (255, 255, 255))

    # Compute CLIP
    gt_image_features = compute_clip(gt_image)
    pred_image_features = compute_clip(pred_image)
    image_empty_features = compute_clip(image_empty)
    clip_distance = torch.cosine_similarity(
        gt_image_features, pred_image_features
    ).item()
    clip_empty_distance = torch.cosine_similarity(
        gt_image_features, image_empty_features
    ).item()
    score_clip = (clip_distance - clip_empty_distance) / (1 - clip_empty_distance)

    # Compute SSIM
    ssim_value = compute_ssim(gt_image, pred_image)
    ssim_empty = compute_ssim(gt_image, image_empty)
    score_ssim = (ssim_value - ssim_empty) / (1 - ssim_empty)

    return LatexResults(
        ssim=ssim_value,
        score_ssim=score_ssim,
        clip_distance=clip_distance,
        score_clip=score_clip,
    )


if __name__ == "__main__":
    latex_code1 = r"""
    \documentclass{article}

\usepackage{graphicx}  % For including images
\usepackage{enumitem}  % For customizing itemized lists
\usepackage{titlesec}  % For customizing section titles
\usepackage{lipsum}    % For generating placeholder text (remove in final document)
\usepackage{amsmath}
% Custom section formatting
\titleformat{\section}
{\normalfont\large\bfseries}
{}
{0em}
{}[\titlerule]

\renewcommand{\thesection}{Problem \arabic{section}}

\newcommand{\problemset}[1]{\section{#1}}

% Custom itemized list formatting
\setlist[enumerate,1]{label=(\roman*)}

\begin{document}
\title{CS149 - Programming Assignment \#4}
\author{Josselin Somerville Roberts, Yoni Gozlan}
\date{\today}
\maketitle

\problemset{Warm-Up: Accessing Tensors (3 Points)}

\begin{enumerate}
  \item A 4D tensor is laid out in memory first batched by the batch dimension, then the z, then the y and finally the x. This is efficient as often any operation performed on the tensor will be performed for the entire batch. Therefore if you want to access a specific x,y,z you will want to do that for the entire batch, and these B elements are going to be contiguous in memory.

\end{enumerate}

\problemset{Part 1: A Simple (But Not So Efficient) Implementation of Attention (10 Points)}

\end{document}
    """

    latex_code2 = r"""
    \documentclass[12pt]{article}

\title{CS149 - Programming Assignment \#4}
\author{Josselin Somerville Roberts, Yoni Gozlan}
\date{November 30, 2023}

\begin{document}

\maketitle

\section*{Warm-Up: Accessing Tensors (3 Points)}
\begin{enumerate}
    \item[(i)] A 4D tensor is laid out in memory first batched by the batch dimension, then the \( z \), then the \( y \) and finally the \( x \). This is efficient as often any operation performed on the tensor will be performed for the entire batch. Therefore if you want to access a specific \( x,y,z \) you will want to do that for the entire batch, and these \( B \) elements are going to be contiguous in memory.
\end{enumerate}

\section*{Part 1: A Simple (But Not So Efficient) Implementation of Attention (10 Points)}

\end{document}
    """

    latex_code3 = r"""
    \documentclass[12pt]{article}

% To create a horizontal line after the title
\newcommand{\sectionline}{%
  \nointerlineskip \vspace{\baselineskip}%
  \hspace{\fill}\rule{\linewidth}{.7pt}\hspace{\fill}%
  \par\nointerlineskip \vspace{\baselineskip}
}

\title{CS149 - Programming Assignment \#4}
\author{Josselin Somerville Roberts, Yoni Gozlan}
\date{November 30, 2023}

\begin{document}

\maketitle

\section*{Warm-Up: Accessing Tensors (3 Points)}
\sectionline
\begin{enumerate}
    \item[(i)] A 4D tensor is laid out in memory first batched by the batch dimension,
    then the \( z \), then the \( y \) and finally the \( x \). This is efficient as often any
    operation performed on the tensor will be performed for the entire batch.
    Therefore if you want to access a specific \( x,y,z \) you will want to do that
    for the entire batch, and these \( B \) elements are going to be contiguous in
    memory.
\end{enumerate}

\section*{Part 1: A Simple (But Not So Efficient) Implementation of Attention (10 Points)}
\sectionline

\end{document}

    """

    latex_code4 = r"""
    \documentclass{article}

\usepackage{graphicx}  % For including images
\usepackage{enumitem}  % For customizing itemized lists
\usepackage{titlesec}  % For customizing section titles
\usepackage{lipsum}    % For generating placeholder text (remove in final document)
\usepackage{amsmath}
% Custom section formatting
\titleformat{\section}
{\normalfont\small\bfseries}
{}
{0em}
{}[\titlerule]

\renewcommand{\thesection}{Problem \arabic{section}}

\newcommand{\problemset}[1]{\section{#1}}

% Custom itemized list formatting
\setlist[enumerate,1]{label=(\roman*)}

\begin{document}
\title{CS149 - Programming Assignment \#4}
\author{Josselin Somerville Roberts, Yoni Gozlan}
\date{\today}
\maketitle

\problemset{Warm-Up: Accessing Tensors (3 Points)}

\begin{enumerate}
  \item A 4D tensor is laid out in memory first batched by the batch dimension, then the z, then the y and finally the x. This is efficient as often any operation performed on the tensor will be performed for the entire batch. Therefore if you want to access a specific x,y,z you will want to do that for the entire batch, and these B elements are going to be contiguous in memory.

\end{enumerate}

\problemset{Part 1: A Simple (But Not So Efficient) Implementation of Attention (10 Points)}

\end{document}
    """

    results = evaluate_latex(latex_code1, latex_code2)
    debug(results)
