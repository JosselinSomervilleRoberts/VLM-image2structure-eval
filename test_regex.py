import re


def find_latex_equations(latex_code):
    # Patterns for different types of equations
    patterns = [
        r"\\\[([\s\S]*?)\\\]",  # \[ ... \]
        r"\\\(([\s\S]*?)\\\)",  # \( ... \)
        r"\$\$([\s\S]*?)\$\$",  # $$ ... $$
        r"\$(.*?)\$",  # $ ... $
        r"\\begin\{equation\}([\s\S]*?)\\end\{equation\}",  # \begin{equation} ... \end{equation}
        r"\\begin\{displaymath\}([\s\S]*?)\\end\{displaymath\}",  # \begin{displaymath} ... \end{displaymath}
        r"\\begin\{math\}([\s\S]*?)\\end\{math\}",  # \begin{math} ... \end{math}
    ]
    patterns_to_remove_from_equations = [
        r"\\label\{.*?\}",  # \label{...}
        r"\\tag\{.*?\}",  # \tag{...}
        r"\\cite\{.*?\}",  # \cite{...}
        r"\\ref\{.*?\}",  # \ref{...}
        r"\\eqref\{.*?\}",  # \eqref{...}
        r"\\newcommand\{.*?\}",  # \newcommand{...}
        r"\\renewcommand\{.*?\}",  # \renewcommand{...}
        r"\\def\{.*?\}",  # \def{...}
        r"\\let\{.*?\}",  # \let{...}
        # " ",  # Remove spaces
        # "\n",  # Remove new lines
        # "\t",  # Remove tabs
    ]

    original_equations = []
    filtered_equations = []

    for pattern in patterns:
        matches = re.findall(pattern, latex_code)
        original_equations.extend(matches)
        for match in matches:
            # Remove unnecessary patterns from equations
            for pattern_to_remove in patterns_to_remove_from_equations:
                match = re.sub(pattern_to_remove, "", match)
            filtered_equations.append(match)

    return original_equations, filtered_equations


def parse_latex(latex_code):
    # Patterns for specific commands to extract text from
    command_patterns = [
        r"\\section\*?\{([^\}]+)\}",
        r"\\subsection\*?\{([^\}]+)\}",
        r"\\subsubsection\*?\{([^\}]+)\}",
        r"\\title\{([^\}]+)\}",
        r"\\textbf\{([^\}]+)\}",
    ]

    # Extract text from specific commands
    extracted_text = []
    for pattern in command_patterns:
        matches = re.findall(pattern, latex_code)
        extracted_text.extend(matches)

    # Reuse the equation parsing function to exclude equations
    original_equations, filtered_equations = find_latex_equations(latex_code)
    for eq in original_equations:
        latex_code = latex_code.replace(eq, "")

    # Regular expression patterns to identify and exclude LaTeX commands and environments
    command_pattern = r"\\[a-zA-Z]+\*?\{.*?\}"
    environment_pattern = r"\\begin\{.*?\}.*?\\end\{.*?\}"

    # Remove commands and environments
    latex_code = re.sub(command_pattern, "", latex_code)
    latex_code = re.sub(environment_pattern, "", latex_code, flags=re.DOTALL)

    # Extract regular text
    # Split by new lines and spaces, filter out empty strings
    text_segments = filter(None, re.split(r"\s+|\n", latex_code))
    extracted_text.extend(text_segments)

    return list(extracted_text), filtered_equations, None


# Example LaTeX code
latex_code = """
\\documentclass{article}
\\usepackage{amsmath}
\\begin{document}

\\section*{Convolution Layer in Computer Vision}

\\textbf{General Convolution Operation:}
\\begin{equation}
    \\text{Output}(i, j) = \\sum_m \\sum_n \\text{Input}(i + m, j + n) \\cdot \\text{Kernel}(m, n)
\\end{equation}

\\textbf{Sobel Filter Example:}

Sobel filter kernels for edge detection:
\\begin{equation}
    G_x = \\begin{bmatrix} -1 & 0 & +1 \\\\ -2 & 0 & +2 \\\\ -1 & 0 & +1 \\end{bmatrix}, \\quad G_y = \\begin{bmatrix} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ +1 & +2 & +1 \\end{bmatrix}
\\end{equation}

Using Sobel filter in convolution:
\\begin{equation}
    \\text{Output}_{\\text{Sobel}}(i, j) = \\sum_m \\sum_n \\text{Input}(i + m, j + n) \\cdot G_x \\text{ or } G_y
\\end{equation}

\\end{document}
"""

text, equations, images = parse_latex(latex_code)
print("Text:", text)
print("Equations:", equations)
print("Images:", images)
