# 300W Facial Landmarks Dataset - Download & Conversion Tool

**Complete guide to download and convert the 300W dataset for facial landmark detection with 68 landmarks format compatible with OpenCV, dlib, and other training frameworks.**

## 📋 Overview

The 300W (300 Faces in-the-Wild) dataset is one of the most popular facial landmark datasets for computer vision research. This repository provides scripts to:

- ✅ Download 300W dataset from Activeloop Hub (free, no registration required)
- ✅ Convert format from `(204,1)` to `(68,2)` landmarks 
- ✅ Generate `.pts` files compatible with training pipelines
- ✅ Ready for facial landmark detection model training

## 🎯 Output

After running this script, you will get:
- **599 high-quality images** with varying resolutions (200×200 to 3000×2000)
- **599 landmark files** in standard `.pts` format
- **68 facial landmarks** per image (x,y coordinates in pixels)
- **100% success rate** conversion
- **Training-ready format** for PyTorch, TensorFlow, or other frameworks

## 📊 Dataset Information

- **Total samples**: 599 images with 68 landmarks each
- **Image sizes**: Variable (optimal for data augmentation)
- **Landmark format**: Standard 68-point facial landmarks
- **File size**: ~2.5 GB total
- **Quality**: High-resolution images from various lighting and pose conditions
- **Use cases**: Perfect for crowd surveillance, facial analysis, emotion detection

## 💾 Requirements

Create `requirements.txt`:

```txt
deeplake>=3.8.0
opencv-python>=4.5.0
numpy>=1.21.0
tqdm>=4.62.0
Pillow>=8.3.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 How to Run

### 1. Clone/Download Repository
```bash
git clone <repository-url>
cd 300w-dataset-converter
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Download Script
```bash
python download_300w.py
```

### 4. Wait for Completion
The script will:
- Download dataset from Activeloop (~5-10 minutes depending on connection)
- Convert landmark format automatically
- Validate conversion results
- Save to `300W_FIXED/` folder

### 5. Verify Results
Check output folder structure:
```
300W_FIXED/
├── images/
│   ├── img_00000.jpg
│   ├── img_00001.jpg
│   └── ... (599 files)
└── pts/
    ├── img_00000.pts
    ├── img_00001.pts
    └── ... (599 files)
```



## 💻 System Requirements

- **Python**: 3.7+
- **RAM**: Minimum 4GB (recommended 8GB+)
- **Storage**: 5GB free space
- **Internet**: Stable connection for download
- **OS**: Windows/Linux/macOS

## 🔍 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'deeplake'"
```bash
pip install --upgrade deeplake
```

### Error: "Permission denied"
```bash
# Windows: Run as Administrator
# Linux/Mac: 
chmod +x download_300w.py
```

### Error: "Insufficient storage"
```bash
# Free up space, need minimum 5GB
# Or change output directory in script
```

## 📚 Citation

If you use this tool in your research, please cite the original 300W dataset:

```bibtex
@inproceedings{sagonas2016300,
  title={300 faces in-the-wild challenge: Database and results},
  author={Sagonas, Christos and Antonakos, Epameinondas and Tzimiropoulos, Georgios and Zafeiriou, Stefanos and Pantic, Maja},
  booktitle={Image and vision computing},
  volume={47},
  pages={3--18},
  year={2016},
  publisher={Elsevier}
}
```

## 📄 License

The 300W dataset follows the original license from Imperial College London. This conversion script is for educational/research purposes.

---

**Simple tool for converting 300W dataset to standard 68-landmark format.**
