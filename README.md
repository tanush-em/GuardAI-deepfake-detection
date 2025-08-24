# 🛡️ GuardAI - Advanced Deepfake Detection

A comprehensive deepfake detection system with advanced forensics capabilities, built with PyTorch and Streamlit.

## 🚀 Features

### Core Detection
- **Real-time Video Analysis**: Process videos with live confidence updates
- **Multiple Model Support**: Choose from various pre-trained models
- **Advanced Analytics**: Comprehensive visualizations and metrics

### 🔍 Fake Region Annotator (NEW!)
- **Facial Region Detection**: Identifies suspicious areas using 68-point facial landmarks
- **Artifact Analysis**: Detects compression, lighting, edge, and texture artifacts
- **Visual Annotations**: Color-coded bounding boxes with confidence scores
- **Heatmap Generation**: Concentration maps of suspicious regions
- **Forensics Reports**: Detailed analysis for legal and investigative purposes

### Additional Features
- **Comprehensive Reporting**: Generate detailed PDF and JSON reports
- **Analytics Dashboard**: Track performance and analyze trends
- **Model Management**: Compare and fine-tune detection models
- **Dark Theme UI**: Modern, professional interface

## 🎯 Use Cases

- **Law Enforcement**: Digital forensics and evidence analysis
- **Content Verification**: Social media and news media verification
- **Research & Development**: Deepfake detection algorithm development
- **Media Forensics**: Professional content authenticity verification

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/GuardAI-deepfake-detection.git
   cd GuardAI-deepfake-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv guardai_env
   source guardai_env/bin/activate  # On Windows: guardai_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📖 Usage

### Basic Video Analysis
1. Navigate to "🎥 Single Video Analysis"
2. Upload a video file (MP4, AVI, MOV, MKV)
3. Configure analysis parameters
4. Click "Start Analysis" to get results

### Fake Region Annotator
1. Navigate to "🔍 Fake Region Annotator"
2. Configure forensics analysis settings
3. Upload video for detailed region analysis
4. Review annotated frames, heatmaps, and reports



## 🔬 Technical Details

### Architecture
- **Backend**: PyTorch with ResNeXt50 + LSTM architecture
- **Frontend**: Streamlit with custom dark theme
- **Computer Vision**: OpenCV and face_recognition
- **Visualization**: Plotly and Matplotlib

### Models
- Multiple pre-trained models with varying accuracy (84%-97%)
- Configurable sequence lengths (10-100 frames)
- Support for different datasets (Celeb-DF, FaceForensics++)

### Fake Region Detection
- **68-point facial landmarks** for precise region identification
- **5 key regions**: Eyes, Nose, Mouth, Face Boundary, Cheeks
- **Artifact analysis**: Compression, lighting, edge, and texture analysis
- **Risk scoring**: Weighted combination of region and artifact scores

## 📊 Performance

- **Accuracy**: Up to 97% on benchmark datasets
- **Speed**: ~2-5 seconds per frame for region analysis
- **Memory**: Moderate usage with configurable parameters
- **Scalability**: Supports multiple models and real-time processing

## 📁 Project Structure

```
GuardAI-deepfake-detection/
├── app.py                      # Main Streamlit application
├── utils.py                    # Utility functions and Fake Region Annotator
├── requirements.txt            # Python dependencies
├── trained-models/             # Pre-trained detection models
├── test-vids/                  # Test video files
├── test_fake_region_annotator.py  # Test script for Fake Region Annotator
├── FAKE_REGION_ANNOTATOR.md    # Detailed documentation
└── README.md                   # This file
```

## 🧪 Testing

Run the test suite to verify functionality:

```bash
python test_fake_region_annotator.py
```

## 📚 Documentation

- **[Fake Region Annotator Guide](FAKE_REGION_ANNOTATOR.md)**: Comprehensive documentation
- **[Features Summary](FEATURES_SUMMARY.md)**: Detailed feature descriptions
- **[Dataset Information](datasetINFO.md)**: Information about supported datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is designed for legitimate forensic and research purposes. Please use responsibly and in compliance with applicable laws and regulations.

## 📞 Support

For questions and support:
- Check the documentation files
- Open an issue on GitHub
- Review the test scripts for usage examples

---

**GuardAI** - Advanced Deepfake Detection with Forensics Capabilities 🛡️
