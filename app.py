import streamlit as st
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
import face_recognition
import os
from torch import nn
import tempfile
import glob

# Model definition (from notebook)
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = torchvision.models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# Dataset for extracting face-cropped frames from video
class ValidationDataset(Dataset):
    def __init__(self, video_path, sequence_length=20, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.count = sequence_length
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(self.video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        if len(frames) == 0:
            raise ValueError("No faces detected in the video.")
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = True
        while success:
            success, image = vidObj.read()
            if success and image is not None:
                yield image

def predict(model, img):
    sm = nn.Softmax(dim=1)
    fmap, logits = model(img)
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return int(prediction.item()), confidence

def main():
    st.title("Deepfake Detection App")
    st.write("Upload a video and select a model to detect if it's REAL or FAKE.")

    # Model selection
    model_files = glob.glob(os.path.join("trained-models", "*.pt"))
    model_names = [os.path.basename(f) for f in model_files]
    model_choice = st.selectbox("Select a model", model_names)

    # Video upload
    uploaded_file = st.file_uploader("Upload a video file (mp4)", type=["mp4"]) 
    if uploaded_file is not None and model_choice:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            tmpfile.write(uploaded_file.read())
            video_path = tmpfile.name
        st.video(video_path)

        # Preprocessing
        im_size = 112
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model(2).to(device)
        model_path = os.path.join("trained-models", model_choice)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        try:
            dataset = ValidationDataset(video_path, sequence_length=20, transform=train_transforms)
            with torch.no_grad():
                frames = dataset[0].to(device)
                pred, conf = predict(model, frames)
            label = "REAL" if pred == 1 else "FAKE"
            st.success(f"Prediction: {label} (Confidence: {conf:.2f}%)")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main() 