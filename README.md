# Action Recognition using Vision Transformers

## Overview

This project implements action recognition in videos using Vision Transformers (ViTs), specifically the TimeSformer model. The model is trained on the HMDB dataset and achieves high accuracy in classifying human actions from video input.

## Key Features

- Vision Transformer-based video action recognition
- Pre-trained TimeSformer model fine-tuning
- Supports multiple action categories from HMDB dataset
- Achieves up to 98.40% top-1 accuracy
- Includes visualization tools for model analysis

## Requirements

- Python 3.x
- PyTorch
- Transformers library
- ffmpeg
- TensorBoard
- scikit-learn
- pandas
- numpy

## Dataset Preparation

1. **Preprocessing Pipeline**:

   - Resize frames to 224x224 pixels
   - Convert images to video format using ffmpeg
   - Organize videos into class-based directories
   - Generate class_path.csv with file paths and labels
   - Create label_mapping.json for class encoding
2. **Dataset Split**:

   - Training: 80%
   - Validation: 10%
   - Test: 10%

## Model Architecture

- Base Model: TimeSformer
- Pre-trained on: Kinetics-400 dataset
- Features:
  - Divided space-time attention
  - Separate handling of spatial and temporal dimensions
  - Transformer-based architecture

## Training Configuration

- Batch Size: 8 (optimal)
- Learning Rate: 0.005
- Optimizer: SGD with momentum 0.9
- Weight Decay: 1e-4
- Dropout Rate: 0.5
- Loss Function: Cross-entropy
- Evaluation Frequency: Every 5 epochs

## Best Performance

- Top-1 Accuracy: 97.60% (4 batch size, 20 epochs)
- Top-5 Accuracy: 100.00%

## Usage

1. **Setup Environment**:

   ```bash
   pip install -r requirements.txt
   ```
2. **Preprocess Dataset**:

   ```bash
   python preprocess_data.py --input_dir /path/to/hmdb --output_dir /path/to/output
   ```
3. **Train Model**:

   ```bash
   python train.py --data_path /path/to/processed_data --batch_size 8 --epochs 20
   ```
4. **Evaluate Model**:

   ```bash
   python evaluate.py --model_path /path/to/checkpoint --test_data /path/to/test_set
   ```

## Visualization

- TensorBoard integration for:
  - Training/validation loss curves
  - Accuracy metrics
  - Confusion matrices
  - Prediction visualizations

## Results

The model shows strong performance across various action categories:

- High accuracy in distinguishing similar actions
- Robust performance on unseen test data
- Consistent performance across different action types

## Known Limitations

- Training instability in loss curves
- Occasional confusion between similar actions (e.g., fencing and draw sword)
- Compute-intensive training process

## Future Improvements

- Implementation of learning rate scheduling
- Enhanced regularization techniques
- Weighted loss functions for class imbalance
- Multi-stream approach integration
- Additional data augmentation methods

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{bertasius2021space,
  title={Is Space-Time Attention All You Need for Video Understanding?},
  author={Gedas Bertasius and Heng Wang and Lorenzo Torresani},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

## License

Acknowledgments

Copyright 2024 [Project Contributors]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

- Project TA: Anindya Mondal
- Contributors:
  - Mohamed Faheem Thanveer (6823682)
  - Mahabaleshwar Poorvita (6801697)
  - Shrinit Sanjiv Patil (6816353)
  - Priyanka Kamila (6787345)
