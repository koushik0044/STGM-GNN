# Spatio-Temporal Graph Neural Network (STGM) for Hydrology Systems

Welcome to the repository for the Spatio-Temporal Graph Model (STGM), a neural network designed to leverage the power of Graph Convolutional Networks (GCNs) and Temporal Convolutional Networks (TCNs) for the integration of static and dynamic features within graph-structured data.

## Overview

This project focuses on developing predictive models for time series data in network environments, specifically predicting depth and inflow features at various nodes in a hydrology system. The project implements two main models: the Graph Time Series Model (STGM) and the Simple Multi-Layer Perceptron (SimpleMLP).

### Features

- **Spatio-Temporal Graph Model (STGM)**: Integrates GCNs and TCNs to process static node features alongside dynamic time series data.
- **Simple Multi-Layer Perceptron (SimpleMLP)**: Processes each node’s time series data independently.
- **Data Preprocessing**: Adjusts adjacency matrix, preprocesses static and dynamic features, and structures datasets for training.
- **Model Training and Evaluation**: Uses RMSE metrics to assess model performance.

## How to Run

To run the STGM model on Google Colab, follow these instructions:

### Step 1: Access the Notebook

The notebook `AI_Project.ipynb` is designed to be run on Google Colab, providing a free GPU and an easy-to-use platform.

- Open the notebook in Google Colab directly from this repository.

### Step 2: Set Up Google Colab

- In Google Colab, set your runtime to use a Python 3 environment with GPU support by selecting `Runtime > Change runtime type` from the menu.

### Step 3: Install Dependencies

- Run the first cell of the notebook to install any required dependencies. This should be done automatically by the notebook.

### Step 4: Google Drive Integration

- If your datasets are stored in Google Drive, you can mount your drive within the Colab notebook using the provided cell code.
- Change the `DATASET_PATH` accordingly.

### Step 5: Execute the Notebook

- Run each cell in the notebook in sequence. You can run a cell by clicking on it and pressing the play button or using the shortcut `Shift + Enter`.

### Step 6: View Results

- After running the notebook, the training progress and validation results will be displayed. The notebook also includes cells for visualizing the model's performance.

## Project Structure

- `AI_Project.ipynb`: Jupyter Notebook detailing the implementation, training, and evaluation of the STGM.
- `AI_Project_Group10.pdf`: Report detailing the project, including background, methodology, and results.
- `ReadMe.txt`: Initial ReadMe file with basic instructions.

## Model Architecture

Here is the architecture of the STGM:

![Model Architecture](AI-FINAL-PROJECT/STGM-GNN/Group 1.jpg)

### Data Preprocessing

- Adjust the adjacency matrix to match the dataset’s node count.
- Preprocess static node features to eliminate non-numeric attributes.
- Convert dynamic time series data into sequential formats.
- Structure the processed sequences and static features into datasets for efficient batch processing.

### Model Architecture

- **Graph Convolutional Network (GCN)**: Processes static features through graph convolutions.
- **Temporal Convolutional Networks (TCNs)**: Capture temporal patterns of dynamic features.
- **Output Layers**: Generate predictions for each node in the hydrology system.

### Training and Evaluation

- **Training**: Iterative process using RMSELoss function and Adam optimizer.
- **Validation**: Evaluate model performance on unseen data to prevent overfitting.
- **Testing**: Assess predictive performance using metrics such as RMSE and MAE.

### Experiments

- **Datasets**: Consist of static features and dynamic time series data, divided into training, validation, and test sets.
- **Baselines**: Compare STGM with a Simple MLP model.
- **Evaluation Results**: Quantitative indicators of model performance, enabling comparison with baseline models.

## Contributors

- **Sai Koushik Katakam** - Texas A&M University - Corpus Christi - [Email](mailto:skatakam1@islander.tamucc.edu)
- **Shivani Jilukara** - Texas A&M University - Corpus Christi - [Email](mailto:sjilukara@islander.tamucc.edu)
- **Thirumala Devi Kola** - Texas A&M University - Corpus Christi - [Email](mailto:tkola1@islander.tamucc.edu)
- **NabilAbdelaziz FerhatTaleb** - Texas A&M University - Corpus Christi - [Email](mailto:nferhattaleb@islander.tamucc.edu)

## References

1. Gao, L., et al. "Spatial-temporal graph neural networks for river flow forecasting." *Water Resources Research*, 2020.
2. Galavi, H., et al. "Prediction of water quality parameters using graph neural networks." *Environmental Science and Pollution Research*, 2021.
3. Yue, X., et al. "Soil moisture prediction using graph neural networks." *Remote Sensing of Environment*, 2022.
4. Zheng, S., et al. "Hydro-graph neural networks for water flow forecasting." *Water Resources Management*, 2023.
5. Rui, X., et al. "Modeling and forecasting river flow using spatial-temporal graph neural networks." *Journal of Hydrology*, 2021.

For more detailed information, please refer to the `AI_Project_Group10.pdf` report.

---

Feel free to reach out with any questions or feedback!
