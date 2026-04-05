# Elite Squad

A **Physics-Informed Deep Learning Model** for accurate air pollution forecasting using hybrid AI + scientific principles.

---

## Problem Statement

Air pollution forecasting (PM2.5) is highly complex due to:
- Spatio-temporal dependencies  
- Weather influence (wind, rain, temperature)  
- Physical transport processes  

Traditional ML models:
- Treat it as simple time-series ❌  
- Fail in real-world dynamic conditions ❌  

---

## Our Solution

We propose a **Hybrid Physics + AI Model** that combines:

-  Deep Learning (Neural Networks)
-  Advection-Diffusion Physics
-  Global Attention Mechanism

This allows the model to:
- Capture long-range dependencies  
- Simulate pollutant transport  
- Improve generalization across environments  

---

##  Model Architecture

### 🔹 Key Components

1. **Feature Extraction Layer**
   - CNN-based spatial learning  

2. **Physics Module**
   - Advection-Diffusion modeling using wind vectors  

3. **Global Attention**
   - Transformer-style learning for long-range dependencies  

4. **Reconstruction Layer**
   - Converts learned features into PM2.5 predictions  

---


## Dataset

Multi-variable atmospheric dataset including:
- PM2.5  
- Temperature  
- Wind (U, V)  
- Rain  
- Pressure  
- Emissions  

Training conducted across multiple months:
- April  
- July  
- October  
- December  

---

## How to Run

### 🔹 1. Clone Repository

```bash
git clone https://github.com/jc-kirthi/Elite-Squad.git
cd Elite-Squad

```
### 🔹2. Install Dependencies

```bash
Install Dependencies
```
### 🔹3. Run Training

```bash
3. Run Training
```
### 🔹4. Run Inference

```bash
4. Run Inference
```
## Usage (Load Model):
```bash
import torch
from model import GeniusChildNet

model = GeniusChildNet()
model.load_state_dict(torch.load("best_physics_model.pt"))
model.eval()
```
## Output
Generates PM2.5 predictions

Saved as:
```bash
preds.npy
```
## Key Innovations

- Physics-informed AI (not pure ML)
- Advection-based pollutant transport
- Global attention for spatial reasoning
- Robust multi-month generalization

## Model Checkpoint

We provide trained model weights for reproducibility:

Hugging Face: https://huggingface.co/jc-kirthi/genius-child-pm25

## License

This project follows ANRF Open License as per hackathon guidelines.

## Future Improvements
- Real-time forecasting system
- Web dashboard integration
- Multi-city deployment
