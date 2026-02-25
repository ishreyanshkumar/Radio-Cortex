# Gradio Deployment Guide

This document explains what folders and files are needed to run `gradio_app.py`.

## 1. Web Files (The UI)
-   **Folder**: `ui/`
-   **What it does**: This holds the custom website code (`index.html`, `dashboard.js`, etc.).
-   **How it connects**: `gradio_app.py` serves this entire folder directly to the browser at the `/ui` URL, which allows the Gradio tabs to embed the custom HTML dashboards.

## 2. Data Files (The Results)
-   **Folder**: `results/`
-   **What it does**: Holds the raw output from simulations (like `experiment_results.csv` and `simulation_data.db`).
-   **How it connects**: `gradio_app.py` serves this folder to the browser at the `/results` URL so that the Javascript code in the `ui/` folder can download and plot the data.

## 3. Analysis Files (The AI Brain)
-   **Folder**: `bdh_results/`
-   **What it does**: Holds the JSON files (like `monosemanticity.json`, `sparsity.json`, `saliency.json`) that explain how the RL agent makes decisions.
-   **How it connects**: `gradio_app.py` imports a script called `interpretability/gradio_tab.py`. That script reads the JSON files directly from the `bdh_results/` folder to draw the "🧠 BDH Interpretability" tab.

## 4. Models (The AI Weights)
-   **Folder**: `models/`
-   **What it does**: Holds the `.pt` checkpoint files (the trained AI memory).
-   **How it connects**: `gradio_app.py` scans this folder to build the dropdown menus for the "Model Inference" and "Model Comparison" tabs.

## 5. Other Important Python Files
To run without crashing, `gradio_app.py` relies on these scripts being present:
-   `policies.py`: Rebuilds the Neural Network architecture when a model is selected.
-   `oran_ns3_env.py`: Provides the environment configuration (like the number of users) the models expect to see.
