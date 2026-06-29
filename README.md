# underspecification-tool-senior-project-bsc-2027
  This repository includes a study on underspecification analysis tool (expected to graduate in 2027). 
  This works is focusing on a creating a tool that can analyze the underspecification of models and show the Underspecification Degree accoring to the [paper].(https://www.scitepress.org/Papers/2025/133742/133742.pdf)

# Key Features
- Using LIME to analyze Underspecification of the models.
  - Show the Original Image, Positive masked region, Negative masked region of the models. (means what is model accually see).
  - Compute the Underspecification Degree between models using Cosine Similarity of LIME masks in Instance and Class level.

# Directory Structure
  (Running the project first time will create these folder automatically)
```text
├── models/      # Put your full trained models (.h5) here
├── images/      # Put your test images here
├── labels/      # Put class label text files here (optional)
└── results/     # Output folder for generated plots and reports (auto-created)
``` 
## labels.txt format
  ```
cat # class 0
dog # class 1
....
```

# Getting Started

## 1. Environment Setup
- Python version: `3.10.11`
- Install key dependencies:
  ```bash
  pip install tensorflow lime scikit-image matplotlib pillow rich tqdm langchain-ollama ollama
  ```
  *(Note: For Apple Silicon Macs, you may use `tensorflow-macos` and `tensorflow-metal`)*

## 2. Ollama Setup
  - Ollama version : `0.30.11`
  - Run this command to pull the model.
  ```ollama pull gemma4:12b```

# Usage

## Running
  Ensure that you are in the project folder then, run `LIMEV9.py` file.
  (Run within setup environmment)
```bash
python LIMEV9.py \
  --models models/xception.h5 models/resnet.h5 \
  --images images/dog.avif images/cat.jpg \
  --labels labels/xception_labels.txt labels/resnet_labels.txt
```
  Note : on the first run without requied folder the code will create the folder automatically and stop. You need to put your models, images in the folder and run again.

## CLI Arguments
| Argument | Type | Required | Description | Mutiple Input |
| :--- | :--- | :---: | :--- |:---:|
| `--models` | paths | **Yes** | Paths to the models to analyze (e.g. `models/model1.h5 models/model2.h5`). | **Yes** |
| `--images` | paths | **Yes** | Paths to the test images (e.g. `images/img1.jpg images/img2.png`). | **Yes** |
| `--labels` | paths | No | Paths to label files. If not provided, defaults to class indexes. | **Yes** |

# Core Loop Description
  - LIME finds the top feature(mask) by `LIMEExplainer` 
  - Overlay the image with mask and show the visualization with comaparison with original.
  - Image Captioning by LLM(Gemma4:12b) for each images.
  - Compute and show the Underspecification Degree, Lower means low underspecification, higher means high underspecification.
  

# Example Data
You can try download example data from [HERE](https://drive.google.com/drive/folders/1TUf2LdaMmNP50LnA0a_GZN6mJrFkgEfj?usp=sharing).