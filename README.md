# Joss image2structure's eval

## Latex

### Install
```bash
# Latex
sudo apt install texlive-latex-extra
pip install latex pdf2image

# SSIM
pip install scikit-image
pip install opencv-python

# CLIP
pip install ftfy regex tqdm 
pip install git+https://github.com/openai/CLIP.git
```

### Run
```bash
python test_latex.py 
```

## React

### Install
Install Node.js and npm, then
```bash
pip install webdriver-manager
pip install selenium
```