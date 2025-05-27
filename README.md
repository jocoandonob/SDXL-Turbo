# SDXL-Turbo Text to Image Web Application

A web application that uses Stability AI's SDXL-Turbo model to generate images from text prompts. Built with Streamlit for a beautiful and interactive user interface.

## Features

- Real-time image generation using SDXL-Turbo
- Clean and intuitive user interface
- Support for negative prompts
- Image download functionality
- Fast inference with optimized settings
- Helpful tips for better results

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- pip (Python package manager)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd SDXL-Turbo
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. The application will automatically open in your default web browser. If it doesn't, navigate to:
```
http://localhost:8501
```

3. Enter your prompt in the text area and click "Generate Image"

## Notes

- The first run will download the SDXL-Turbo model weights (approximately 4GB)
- Image generation requires a CUDA-capable GPU for optimal performance
- The application uses optimized settings for fast inference (1 step, no guidance)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 