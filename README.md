# MNIST Digit Recognition Dashboard

A Streamlit dashboard that recognizes handwritten digits with a trained PyTorch CNN.
It supports webcam capture and image upload.

## Setup

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`.

## Retrain the model

The trained model is stored in `mnist_cnn.pth`.

```powershell
python train.py --epochs 5
```

Optional arguments:

- `--batch-size`
- `--learning-rate`
- `--output`
