# 🛩️ UAVCAN Attack Detection System

A machine learning-based system for detecting fake UAV communications and attacks in UAVCAN (UAV Controller Area Network) protocols.

## 📋 Overview

This project implements a comprehensive attack detection system specifically designed for UAVCAN networks, which are widely used in unmanned aerial vehicles (UAVs/drones) for reliable communication between flight controllers, sensors, and actuators.

### 🎯 Supported Attack Types

- **🌊 Flooding Attacks**: Overwhelming the network with excessive messages
- **🎯 Fuzzing Attacks**: Sending malformed or unexpected data packets  
- **🔄 Replay Attacks**: Retransmitting previously captured legitimate messages

## 🔧 Features

### 🤖 Multiple Detection Models
- **Random Forest**: For flooding and fuzzing detection
- **Neural Network (MLP)**: For replay attack detection
- **Selective Model Loading**: Enable/disable specific models as needed

### 🚀 Processing Methods
- **Cascade Detection**: Sequential model execution for optimized performance
- **Parallel Detection**: Simultaneous model execution for comprehensive analysis

### 🌐 Web Interface
- **Gradio-based GUI**: User-friendly web interface
- **Real-time Results**: Live detection results with detailed metrics
- **Performance Analytics**: Confusion matrix, precision, recall, F1-score

## 📦 Installation

### Prerequisites
```bash
pip install torch torchvision
pip install scikit-learn
pip install pandas numpy
pip install joblib
pip install gradio
```

### Required Model Files
Ensure these trained model files are in your project directory:
- `random_forest_model.pkl` (Flooding detection)
- `random_forest_model_fuzzy.pkl` (Fuzzing detection) 
- `model_27_05_replay.pth` (Replay detection)
- `scaler.pkl` (Feature scaling for neural network)

## 🚀 Usage

### Command Line Interface
```python
from attack_detector import SelectiveAttackDetector

# Enable all models
detector = SelectiveAttackDetector(
    scaler_path='scaler.pkl',
    enable_models={'flooding': True, 'fuzzing': True, 'replay': True}
)

# Process UAVCAN data file
results = detector.process_file('uavcan_data.bin', method='cascade')
```

## 📊 Data Format

The system expects UAVCAN data in the following format:
```
[Label] (timestamp) can_id [length] byte0 byte1 byte2 ...

Example:
Normal (1634567890.123) 0x123 [8] 01 02 03 04 05 06 07 08
Attack (1634567890.456) 0x456 [4] AA BB CC DD
```

## 🏗️ Architecture

### SelectiveAttackDetector Class
- **Model Management**: Dynamic loading/unloading of detection models
- **Feature Engineering**: Automatic extraction of timing and payload features

### CustomMLP Neural Network
```
Input Layer (11 features) → 
Hidden Layer 1 (128 neurons) + BatchNorm + Dropout →
Hidden Layer 2 (64 neurons) + BatchNorm + Dropout →
Hidden Layer 3 (32 neurons) + BatchNorm + Dropout →
Output Layer (2 classes: Normal/Attack)
```

### Feature Set
- `timestamp_diff`: Time difference between consecutive messages
- `can_id`: UAVCAN message identifier
- `length`: Message payload length
- `byte_0` to `byte_7`: Payload bytes (zero-padded)

## 📈 Performance Metrics

The system provides comprehensive performance analysis:

- **Accuracy**: Overall correct prediction rate
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of prediction results

## 🔬 Detection Strategy

### Cascade Mode (Recommended)
1. **Step 1**: Flooding detection on all data
2. **Step 2**: Fuzzing detection on non-flooding data
3. **Step 3**: Replay detection on remaining data

### Parallel Mode
- All models run simultaneously on the entire dataset
- Priority-based result aggregation (Flooding > Fuzzing > Replay)

## ⚙️ Configuration Options

### Model Selection
```python
enable_models = {
    'flooding': True,   # Enable/disable flooding detection
    'fuzzing': False,   # Enable/disable fuzzing detection  
    'replay': True      # Enable/disable replay detection
}
```

### Processing Options
- **Method**: `'cascade'` or `'parallel'`
- **Batch Size**: Configurable for GPU memory optimization
- **Precision**: Timestamp precision for feature engineering

## 🛡️ Security Applications

### UAV/Drone Security
- **Flight Controller Protection**: Detect malicious commands
- **Communication Security**: Monitor inter-component communications


## 📝 Output Examples

### Summary Report
```
📊 DETECTION SUMMARY
Enabled Models: Flooding, Replay
Total Lines Processed: 1000
Correct Predictions: 987
Overall Accuracy: 98.70%
Attacks Detected: 45
Normal Traffic: 955
```
