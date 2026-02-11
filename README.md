# PULSE: Onboard AI Payload for Space Radiation Analysis

## 🛰️ Project Overview
This project aims to develop a lightweight, onboard AI algorithm for CubeSats (Teensy 4.0 MCU) to classify space radiation particles collected by Timepix sensors in real-time.

## 🎯 Key Achievements
- **Model:** Optimized Standard CNN (Gap + BatchNormalization)
- **Performance:** 90.44% Accuracy on VZLUSAT-1 Dataset
- **Size:** 102 KB (Suitable for MCU Flash)
- **Input:** 32x32 Timepix Sensor Data

## 📂 Repository Structure
- `src/`: Python scripts for data processing, training, and validation.
- `models/`: converted TFLite model (.tflite).
- `results/`: Confusion matrices, sample predictions, and benchmark charts.
- `data/`: (Excluded from git) Dataset files.

## 🚀 How to Run
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

