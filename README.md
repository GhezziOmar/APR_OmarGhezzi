# Comparative Analysis of Traditional, Hypercomplex, and Pre-trained Deep Neural Networks for Audio Emotion Recognition

![Python](https://img.shields.io/badge/python-3-blue.svg)
![License](https://img.shields.io/badge/License-GNU%20GPL%20v3-orange.svg)

Demo code for the Audio Pattern Recognition exam project. We introduce novel hypercomplex models, including CliffSER1D, CliffSER2D, PureCliffSER1D and PureCliffSER2D, designed to predict discrete emotional classes from hand-crafted MFCCs and logMel spectrogram features in both 1D and 2D domains.

## Getting Started

### Installation
Download the repository, then:
   
1. Move to the APR_project directory and install the dependencies:
   ```sh
   conda env create -f environment.yml

2. Activate the CliffPhys conda environment:
   ```sh
   conda activate APR

### Training

*Action n*: Run training with 5-fold cv (and Grid Search) one of the n = 0,1,2,3,4,5,6,7.
   ```sh
   python run_GSCV.py -a n
   ```


with: 0 = 'CliffSER1D on EMD data', 1 = 'PureCliffSER1D on EMD data', 2 = 'CliffSER1D', 3 = 'PureCliffSER1D', 4 = 'CliffSER2D on EMD data', 5 = 'CliffSER2D', 6 = 'PureCliffSER2D', 7 = 'PureCliffSER2D on EMD data'

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
