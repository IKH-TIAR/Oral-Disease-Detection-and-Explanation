# Oral Disease Detection and Explanation

## Overview

This repository contains an oral-disease classification and explanation demo that combines a deep learning image classifier with explainable AI (Grad-CAM++) visualizations and an LLM-powered explanation layer presented through a Gradio web UI.

Key ideas:

- Classify intraoral images into one of seven oral conditions.
- Provide visual explanations (Grad-CAM++) that highlight image regions driving the model's prediction.
- Use an LLM to generate human-friendly explanations and guidance for different audiences (Patient or Dentist).

## What it does

- Loads an InceptionResNetV2 model trained for oral-disease classification.
- Accepts images via a Gradio interface.
- Runs prediction and produces a probability histogram for all classes.
- Generates a Grad-CAM++ heatmap to highlight image regions important for the predicted label.
- Calls an LLM to produce explanation text tailored to either a patient or a dentist.

## Model

- Base architecture: InceptionResNetV2 (via the `timm` library).
- Number of classes: 7 (labels in the code: `['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']`).
- The demo expects a saved model weights file (example path used in the code: `/content/drive/MyDrive/datasets/model_teeth_V2.pth`). Update this path for your environment.

## Explainable AI (Grad-CAM++)

- The project uses Grad-CAM++ to produce class-discriminative heatmaps over the input image.
- Grad-CAM++ computes importance weights using higher-order gradients and produces sharper heatmaps than vanilla Grad-CAM in many settings.
- The code hooks into the final convolutional block of InceptionResNetV2 to capture feature maps and gradients, then computes and overlays a colored heatmap on the original image.
- The output helps users (and clinicians) inspect where the model is focusing and increases trust & interpretability.

## Gradio interface features

The `main.py` Gradio app provides:

- Image upload (supports PIL images).
- Audience selection: `Patient` or `Dentist` — this controls the detail level in the generated explanation.
- A `Classify` button that runs prediction, generates Grad-CAM++ overlay, produces an LLM explanation, and saves a prediction histogram.
- A `Clear` button to reset the UI.
- Results tab: shows the Grad-CAM++ image, prediction label with confidence, and a probability histogram.
- Explanation tab: shows the LLM-generated explanation tailored to the selected audience.

## Setup & Requirements

Install dependencies (recommended to use a virtual environment):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision timm pillow opencv-python matplotlib gradio requests
```

## How to run

1. Ensure dependencies are installed and your model weights are available.
2. Set/verify the model path inside `main.py` or update the code to read from a config/env var.
3. Start the Gradio app:

By default the app is launched with `share=True` in the demo; remove or change this to control public sharing.

## Improvements and roadmap

Below are recommended improvements and extension ideas — including how to convert this into a LangChain-based app.

1. LangChain conversion (high-level plan)

   - Goal: Turn the app into a LangChain-driven assistant that can maintain context, handle multi-turn conversation, and optionally use retrieval to augment explanations.
   - Steps:
     a. Separate concerns: create modules for model inference, explanation (Grad-CAM), LLM prompting, and web UI.
     b. Replace direct LLM calls with a LangChain `LLM` wrapper (e.g., OpenAI or Hugging Face LLM wrapper) and use LangChain `PromptTemplate`s for patient vs dentist prompts.
     c. Build a small conversational `Chain` or `Agent` that: (1) accepts an image, (2) returns prediction + gradcam, (3) produces an LLM explanation, and (4) accepts follow-up clarifying questions.
     d. If you want to provide references or past cases, introduce a vector store (FAISS or Chroma) with embeddings (for textual case notes or guidelines) and use LangChain RetrievalQA patterns.
     e. Add tools to the agent (e.g., a `search_guidelines` tool that queries a stored knowledge base for evidence-based treatment steps).
     f. Add caching of LLM outputs and rate-limit or batch calls where possible.

   - Implementation notes:
     - Keep the image processing and Grad-CAM components separate from the LLM chain. LangChain should orchestrate text and metadata, not image pixel math.
     - For multi-modal enhancements, you can store image metadata, prediction, and extracted captions/descriptions in a vector DB to support retrieval-based explanations.
     - Consider building an API (FastAPI/Flask) around the model + Grad-CAM, and let LangChain call this API as a tool.

2. UX and features
   - Add image cropping / brightness adjustment in the UI to help the model when images vary.
   - Provide downloadable reports (PDF) combining the image, heatmap, prediction, and LLM explanation.
   - Add role-based access (clinician vs researcher) and logging/auditing of inference calls.
