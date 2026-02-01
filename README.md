# FastGCN: Breaking the Scalability Bottleneck in Graph Neural Networks

This repository contains Jupyter notebook files to run reproducibility experiments based on the FastGCN paper. FastGCN proposes an efficient batch training strategy for Graph Convolutional Networks (GCNs) by using importance sampling to address the "neighborhood explosion" problem.

---

## 📂 Notebooks Overview

The repository is organized into four primary notebooks, ranging from baseline implementations to experimental extensions:

* **FastGCN_ReproducabilityFull**: 
    A standard GCN implementation used as a baseline to compare performance and accuracy against sampling-based methods.
* **FastGCN_ReproducabilityUniform**: 
    Implementation of FastGCN using **Uniform Sampling**, demonstrating the basic speedups achieved by layer-wise sampling.
* **FastGCN_ReproducabilityImportance**: 
    The core reproducibility experiment for FastGCN using **Importance Sampling** to reduce variance and improve convergence.
* **FastGCN_MultiAttentionExtension**: 
    An experimental extension (FastGAT) that integrates Multi-Head Attention mechanisms into the FastGCN sampling framework.

---

## 🚀 Getting Started


1.  **Environment Setup**:
    Ensure you have a Python environment with Jupyter installed along with the following dependencies:
    * `torch`
    * `torch_geometric`
    * `numpy`
    * `matplotlib`
2.  **Run the experiments**:
    Open any of the `.ipynb` files in Jupyter Lab or Google Colab to begin the training and evaluation process.

---
