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

# Load CLIP
print("Loading CLIP...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print("CLIP loaded")


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


def compute_clip(image: np.ndarray) -> torch.Tensor:
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features


def main(latex_code1: str, latex_code2: str):
    image1 = latex_to_image(latex_code1)
    image2 = latex_to_image(latex_code2)
    image_empty = Image.new("RGB", image1.size, (255, 255, 255))

    image1_features = compute_clip(image1)
    image2_features = compute_clip(image2)
    image_empty_features = compute_clip(image_empty)

    # Convert to numpy array
    image1 = np.array(image1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = np.array(image2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image_empty = np.ones(image1.shape, dtype=np.uint8) * 255
    debug(image1)
    debug(image2)
    debug(image_empty)
    ssim_index = ssim(image1, image2)
    ssim_index_empty = ssim(image1, image_empty)
    score = (ssim_index - ssim_index_empty) / (1 - ssim_index_empty)
    print(f"SSIM index: {ssim_index}")
    print(f"SSIM index empty: {ssim_index_empty}")
    print(f"Score: {score}")

    # Compute CLIP
    debug(image1_features)
    debug(image2_features)
    debug(image_empty_features)
    score = torch.cosine_similarity(image1_features, image2_features).item()
    score_empty = torch.cosine_similarity(image1_features, image_empty_features).item()
    print(f"CLIP similarity: {score}")
    print(f"CLIP similarity empty: {score_empty}")
    score_clip = (score - score_empty) / (1 - score_empty)
    print(f"CLIP Score: {score_clip}")

    # Save image
    cv2.imwrite("image1.png", image1)
    cv2.imwrite("image4.png", image2)


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

    main(latex_code1, latex_code4)
