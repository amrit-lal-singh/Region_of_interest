# Region-of-Interest Confidence Analysis for LLMs

This repository contains the implementation of Region-of-Interest Confidence Analysis (ROICA), a framework for analyzing fine-grained confidence variations within Large Language Model outputs.

## Overview

ROICA decomposes model responses into semantically meaningful regions to analyze confidence variations within a single response. Through this approach, we can detect knowledge boundaries, reasoning inconsistencies, and potential hallucinations with greater precision than global confidence metrics.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/amrit-lal-singh/Region_of_interest.git
cd Region_of_interest
```

2. Set up a Python environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies using uv:
```bash
pip install uv
uv pip install -r requirements.txt
```

## API Keys Setup

This project requires API keys for:

1. **Hugging Face**: Used in `llm.py` to access models
2. **Google Gemini**: Used in `roi.py` for identifying regions of interest

You'll need to add your own API keys:

1. Edit `llm.py` and replace `YOUR_HF_TOKEN_HERE` with your Hugging Face token
2. Edit `roi.py` and replace `YOUR_GEMINI_API_KEY_HERE` with your Gemini API key

## Usage

The project consists of three main components:

1. **LLM Inference**: Generate responses with token-level confidence data
```bash
python llm.py
```

2. **ROI Identification**: Identify regions of interest in responses
```bash
python roi.py
```

3. **ROI Entropy Analysis**: Analyze confidence patterns in identified regions
```bash
python roi_entropy.py
```

## File Structure

- `llm.py`: Handles inference with the Qwen2.5-3B model, generating responses with token probability information
- `roi.py`: Uses Google's Gemini API to identify the region of interest in each response
- `roi_entropy.py`: Analyzes token probabilities within identified regions
- `test_set.json`: Contains test questions (easy and difficult)
- `output_set.json`: Stores model outputs with confidence data

## Research Paper

This implementation is based on research described in `BTP-2_region_of_interest.pdf`, which demonstrates that examining the probability of the first token in answer-specific regions provides a strong signal for hallucination detection.

## License

This project is provided for research purposes. 