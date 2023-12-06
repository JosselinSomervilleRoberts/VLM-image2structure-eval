from typing import Optional, List, Tuple

from latex import build_pdf
from pdf2image import convert_from_bytes
import io
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity as ssim
from toolbox.printing import debug, ldebug
import numpy as np
import cv2
import clip
import torch
from dataclasses import dataclass
import argparse
import os
from tqdm import tqdm
from parser_latex import parse_latex
from process_latex import process_sympy
from nltk.translate.bleu_score import sentence_bleu


@dataclass
class LatexProblem:
    source_code: str
    assets: List[str]
    crop_and_resize: bool = False


@dataclass
class LatexEvalRequest:
    problem: LatexProblem
    generated_code: str


@dataclass
class LatexResults:
    image: Image
    # ssim: float
    # score_ssim: float
    clip_distance: float
    score_clip: float
    sim_text: float
    sim_equations: float
    sim_figures: float


def latex_to_pdf(latex_code: str, assets_path: str) -> io.BytesIO:
    # Compiling LaTeX code to PDF
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), assets_path)
    pdf = build_pdf(latex_code, texinputs=[path, ""])
    return io.BytesIO(pdf.data)  # Convert PDF to a byte stream


def pdf_to_image(
    pdf_stream: io.BytesIO,
    crop: bool = False,
    resize_to: Optional[Tuple[int, int]] = None,
) -> Image:
    # Convert the first page of the PDF stream to an image
    images = convert_from_bytes(pdf_stream.read(), first_page=1, last_page=1)
    if images:
        image: Image = images[0]

        # Removes the white border around the image
        if crop:
            # TODO: Clean this
            # We need to remove the bottom of the image first to remove the number of the page
            image = image.crop(
                (
                    0,
                    0,
                    image.size[0],
                    image.size[1] - int(image.size[1] * 0.13),
                )
            )
            image = image.crop(ImageOps.invert(image).getbbox())

        # Resize the image
        if resize_to:
            image = image.resize(resize_to)

        return image
    else:
        raise Exception("PDF to Image conversion failed")


def latex_to_image(
    latex_code: str,
    assets_path: str,
    crop: bool = False,
    resize_to: Optional[Tuple[int, int]] = None,
):  # -> Tuple[Image, Tuple[int, int]]:
    try:
        pdf_stream = latex_to_pdf(latex_code, assets_path=assets_path)
        image = pdf_to_image(pdf_stream, crop=crop, resize_to=resize_to)
        return image, image.size
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


def evaluate(request: LatexEvalRequest, assets_path: str) -> LatexResults:
    """Evaluate the similarity between two latex codes

    Args:
        request (LatexEvalRequest): Request containing the source code and the generated code
        assets_path (str): Path to the assets

    Returns:
        LatexResults: Results of the evaluation
    """
    # Convert LaTeX to images
    gt_image, gt_size = latex_to_image(
        request.problem.source_code,
        crop=request.problem.crop_and_resize,
        assets_path=assets_path,
    )
    pred_image, pred_size = latex_to_image(
        request.generated_code,
        crop=request.problem.crop_and_resize,
        resize_to=gt_size if request.problem.crop_and_resize else None,
        assets_path=assets_path,
    )
    if request.problem.crop_and_resize:
        assert gt_size == pred_size, f"Expected size {gt_size} but got {pred_size}"
    image_empty = Image.new("RGB", gt_size, (255, 255, 255))

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

    # # Compute SSIM
    # ssim_value = compute_ssim(gt_image, pred_image)
    # ssim_empty = compute_ssim(gt_image, image_empty)
    # score_ssim = (ssim_value - ssim_empty) / (1 - ssim_empty)

    # Compute content similarity
    gt_text, gt_equations, gt_figures = parse_latex(request.problem.source_code)
    pred_text, pred_equations, pred_figures = parse_latex(request.generated_code)
    gt_text = " ".join(gt_text)
    pred_text = " ".join(pred_text)
    sim_text = (
        sentence_bleu([gt_text.split()], pred_text.split(), weights=(0.5, 0.5, 0, 0))
        if len(gt_text) > 0
        else None
    )
    sim_equations = 0
    num_equations = max(len(gt_equations), len(pred_equations))
    for i, eq in enumerate(gt_equations):
        if i < len(pred_equations):
            try:
                gt_sympy = process_sympy(eq)
                try:
                    pred_sympy = process_sympy(pred_equations[i])
                    sim_equations += gt_sympy.equals(pred_sympy)
                    # print(
                    #     f"EQUATION {i}: {gt_sympy} == {pred_sympy} was parsed correctly"
                    # )
                except:
                    # Ground truth equation is a valid sympy equation but the generated equation is not
                    # Compute BLEU score instead
                    # TODO: Think if this should be penalized
                    sim_equations += sentence_bleu(
                        [eq.split()],
                        pred_equations[i].split(),
                        weights=(0.5, 0.5, 0, 0),
                    )
                    # print(
                    #     f"EQUATION {i}: {gt_sympy} was parsed correctly but {pred_equations[i]} was not"
                    # )
            except:
                # Ground truth equation is not a valid sympy equation
                # Compute BLEU score instead
                sim_equations += sentence_bleu(
                    [eq.split()], pred_equations[i].split(), weights=(0.5, 0.5, 0, 0)
                )
                # print(f"EQUATION {i}: {eq} was not parsed correctly")
    sim_equations = sim_equations / num_equations if num_equations > 0 else None
    # Figures should match exactly, create two sets and compute the intersection
    sim_figures = (
        len(set(gt_figures) & set(pred_figures))
        / max(len(set(gt_figures)), len(set(pred_figures)))
        if max(len(set(gt_figures)), len(set(pred_figures))) > 0
        else None
    )

    return LatexResults(
        image=pred_image,
        # ssim=ssim_value,
        # score_ssim=score_ssim,
        clip_distance=clip_distance,
        score_clip=score_clip,
        sim_text=sim_text,
        sim_equations=sim_equations,
        sim_figures=sim_figures,
    )


def generate_prompt(problem: LatexProblem, path: str):  # -> Tuple[str, Image]:
    """Given a problem, generate a prompt for the VLM model.

    Args:
        problem (LatexProblem): Problem to generate a prompt for
        path: Path where the image was generated

    Returns:
        prompt: Prompt for the VLM model
        image: Image generated from the problem
    """

    # Step 1: Generate the prompt
    prompt: str = "Generate the LaTex code to reproduce the following image:\n"
    if len(problem.assets) > 0:
        prompt += f"You are provided with the following assets:\n"
        for asset in problem.assets:
            prompt += f"- {asset}\n"
        prompt += "\n"
    prompt += "You can describe what you are doing but make sure to provide the latex code between $$ tags so that it is rendered correctly."

    # Step 2: Generate the image
    assets_path: str = os.path.join("/".join(path.split("/")[:-1]), "assets")
    image, size = latex_to_image(
        problem.source_code, crop=problem.crop_and_resize, assets_path=assets_path
    )
    image.save(f"{path}.png")

    return prompt, image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "evaluate"],
        help="Mode to run the script in",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="latex_data",
    )
    return parser.parse_args()


if __name__ == "__main__":
    problems: List[LatexProblem] = [
        LatexProblem(
            source_code=r"""\documentclass{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\title{Newton's Binomial Theorem}\begin{document}\maketitle\section{Introduction}Newton's Binomial Theorem is a fundamental theorem in algebra that describes the expansion of powers of a binomial. According to the theorem, it is possible to expand the power \((a + b)^n\) into a sum involving terms of the form \(a^kb^{n-k}\), where the coefficient of each term is a specific positive integer known as a binomial coefficient.\section{The Binomial Theorem}For any positive integer \(n\), the expansion of \((a + b)^n\) is given by:\begin{equation}    (a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k}b^k\end{equation}where \(\binom{n}{k}\) represents the binomial coefficient, calculated as:\begin{equation}    \binom{n}{k} = \frac{n!}{k!(n-k)!}\end{equation}\end{document}""",
            assets=[],
            crop_and_resize=True,
        ),
        LatexProblem(
            source_code=r"""\documentclass{article}\usepackage[utf8]{inputenc}\usepackage{amsmath}\begin{document}\begin{equation}    (a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k}b^k\end{equation}\end{document}""",
            assets=[],
            crop_and_resize=True,
        ),
        LatexProblem(
            source_code=r"""\documentclass{article}\usepackage{graphicx}\usepackage{subcaption}\begin{document}\begin{figure}[h]    \centering    \begin{subfigure}{0.45\textwidth}        \includegraphics[width=\linewidth]{image_0.png}        \caption{Image 0}    \end{subfigure}    \hfill    \begin{subfigure}{0.45\textwidth}        \includegraphics[width=\linewidth]{image_1.png}        \caption{Image 1}    \end{subfigure}    \vspace{1cm}    \begin{subfigure}{0.45\textwidth}        \includegraphics[width=\linewidth]{image_2.png}        \caption{Image 2}    \end{subfigure}    \hfill    \begin{subfigure}{0.45\textwidth}        \includegraphics[width=\linewidth]{image_3.png}        \caption{Image 3}    \end{subfigure}    \vspace{1cm}    \begin{subfigure}{0.45\textwidth}        \includegraphics[width=\linewidth]{image_4.png}        \caption{Image 4}    \end{subfigure}    \hfill    \begin{subfigure}{0.45\textwidth}        \includegraphics[width=\linewidth]{image_5.png}        \caption{Image 5}    \end{subfigure}    \caption{Grid of Images}\end{figure}\end{document}""",
            assets=[f"image_{i}.png" for i in range(6)],
            crop_and_resize=False,
        ),
        LatexProblem(
            source_code=r"""\documentclass{article}\usepackage{graphicx}\usepackage{lipsum}\usepackage{caption}\begin{document}\noindent\begin{minipage}[t]{0.49\textwidth}    \vspace{0pt}    \includegraphics[width=\linewidth]{image_0.png}    \par\vspace{\abovecaptionskip}    \includegraphics[width=\linewidth]{image_1.png}    \par\vspace{\abovecaptionskip}    \captionof{figure}{Caption for both images}\end{minipage}\hfill\begin{minipage}[t]{0.49\textwidth}    \vspace{0pt}    \lipsum[1]\end{minipage}\end{document}""",
            assets=[f"image_{i}.png" for i in range(2)],
            crop_and_resize=True,
        ),
        LatexProblem(
            source_code=r"""\documentclass{article}\usepackage{amsmath}\begin{document}\section*{Convolution Layer in Computer Vision}\textbf{General Convolution Operation:}\begin{equation}    \text{Output}(i, j) = \sum_m \sum_n \text{Input}(i + m, j + n) \cdot \text{Kernel}(m, n)\end{equation}\textbf{Sobel Filter Example:}Sobel filter kernels for edge detection:\begin{equation}    G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}\end{equation}Using Sobel filter in convolution:\begin{equation}    \text{Output}_{\text{Sobel}}(i, j) = \sum_m \sum_n \text{Input}(i + m, j + n) \cdot G_x \text{ or } G_y\end{equation}\end{document}""",
            assets=[],
            crop_and_resize=True,
        ),
    ]

    args = parse_args()

    if args.mode == "generate":
        # Make data path if it doesn't exist
        os.makedirs(args.data_path, exist_ok=True)

        # Generate prompts
        for i, problem in tqdm(
            enumerate(problems), desc="Generating prompts", total=len(problems)
        ):
            prompt, image = generate_prompt(problem, f"{args.data_path}/image_{i}")
            # Save the prompt
            with open(f"{args.data_path}/prompt_{i}.txt", "w") as file:
                file.write(prompt)

    elif args.mode == "evaluate":
        assert os.path.exists(args.data_path), f"Path {args.data_path} does not exist"

        for i, problem in tqdm(enumerate(problems), "Evaluating", total=len(problems)):
            # Read the generated code
            with open(f"{args.data_path}/generated_{i}.txt", "r") as file:
                generated_code = file.read()

            # Evaluate the generated code
            results = evaluate(
                LatexEvalRequest(problem=problem, generated_code=generated_code),
                assets_path=os.path.join(args.data_path, "assets"),
            )
            # Save image
            results.image.save(f"{args.data_path}/generated_{i}.png")
            print(f"\nResults for problem {i}:")
            ldebug(results)
