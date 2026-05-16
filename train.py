import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import av
from tqdm import tqdm

from wave import ViViT
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import logging
from torchvision import transforms
from PIL import Image
import time
from thop import profile
import pandas as pd
import os
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 7e-5
epochs = 200
batch_size = 4
time_size = 16
work = 2
log_interval = 300
height = width = 512

img_size = height  # Since height = width

ROOT_DIR = Path(__file__).resolve().parent
DATA_ROOT = os.environ.get("WAVER_DATA_ROOT", str(ROOT_DIR / "data" / "LaparoClipsP2"))
LOG_DIR = ROOT_DIR / "logs"
WEIGHTS_DIR = ROOT_DIR / "weights"
LOG_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

log_file_name = str(LOG_DIR / "wave.log")
logging.basicConfig(filename=log_file_name, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


label_dict = {
    "AbdominalEntry": 0,
    "Excise": 1,
    "HookCut": 2,
    "Incise": 3,
    "LocPanoView": 4,
    "NeedleIn": 5,
    "NeedleOut": 6,
    "PanoView": 7,
    "ScissorCut": 8,
    "Suction": 9,
    "UseClip": 10
}

num_labels = len(label_dict)

class ClipsDataset(Dataset):
    def __init__(self, data_path, csv_file, transform=None, resize_shape=(width, height)):
        self.data_path = data_path
        self.data_label = pd.read_csv(csv_file)
        self.resize_shape = resize_shape

        if transform is None:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # relative_video_path = self.data_label.iloc[idx, 0].replace("\\", "/")  # Replace backslashes with forward slashes
        relative_video_path = self.data_label.iloc[idx, 1]
        video_path = os.path.join(self.data_path, "clips", relative_video_path)
        label = label_dict[self.data_label.iloc[idx, 2]]


        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format='rgb24')
            img = Image.fromarray(img).resize(self.resize_shape)
            if self.transform:
                img = self.transform(img)

            frames.append(img)

        desired_frames = time_size
        if len(frames) < desired_frames:
            frames += [frames[-1]] * (desired_frames - len(frames))
        elif len(frames) > desired_frames:
            frames = frames[:desired_frames]

        frames = torch.stack(frames)

        return frames, label


def get_input_size_from_dataset(data_loader):
    for x, _ in data_loader:
        return x.shape

def calculate_confusion_matrix(model, data_loader):
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    cm = confusion_matrix(all_targets, all_preds)
    return cm

def calculate_gflops_and_parameters(model, input_size):
    model.eval()
    dummy_input = torch.randn(1, *input_size[1:], device=device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    gflops = flops / 1e9
    return gflops, params

def calculate_inference_speed(model, data_loader):
    model.eval()
    total_time = 0
    num_samples = 0
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            start_time = time.time()
            _ = model(x)
            total_time += time.time() - start_time
            num_samples += 1
    avg_time_per_sample = total_time / num_samples
    return avg_time_per_sample

def get_gpu_memory_usage():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    gpu_index = torch.cuda.current_device()
    allocated_memory = torch.cuda.memory_allocated(gpu_index) / 1024**3
    reserved_memory = torch.cuda.memory_reserved(gpu_index) / 1024**3
    return allocated_memory, reserved_memory

def train():
    train_data_path = DATA_ROOT
    val_data_path = DATA_ROOT
    train_csv_file = os.path.join(train_data_path, "train.csv")
    val_csv_file = os.path.join(val_data_path, "val.csv")

    # Augmentations for training data
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor()  # Ensure this is included in both train and validation transforms
    ])

    # No augmentations for validation data
    val_transform = transforms.Compose([
        transforms.ToTensor()  # Only necessary transformation
    ])

    train_dataset = ClipsDataset(train_data_path, train_csv_file, transform=train_transform)
    val_dataset = ClipsDataset(val_data_path, val_csv_file, transform=val_transform)  # Corrected transform

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=work, pin_memory=True, drop_last=True) # Drop the last batch if it's smaller than batch_size
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=work, pin_memory=True, drop_last=True)

    input_size1 = get_input_size_from_dataset(train_loader)
    input_size2 = get_input_size_from_dataset(val_loader)
    print("Input size1:", input_size1)
    print("Input size2:", input_size2)

    torch.autograd.set_detect_anomaly(True)

    model = ViViT(height, 32, num_labels, time_size)

    model.to(device)

    loss_fc = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_accuracy = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_accuracy = 0, 0
        for step, (x, y) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            # print(f"x shape: {x.shape}, y shape: {y.shape}")  # Add this line
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fc(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_accuracy += (y == torch.argmax(out, dim=-1)).float().mean().item()

            if step % log_interval == 0:
                allocated_memory, reserved_memory = get_gpu_memory_usage()
                log_message = (
                    f"Epoch {epoch} - Step {step} - Train Loss: {loss.item():.4f}, "
                    f"Train Accuracy: {(y == torch.argmax(out, dim=-1)).float().mean().item():.4f}, "
                    f"Allocated memory: {allocated_memory:.2f} GB, Reserved memory: {reserved_memory:.2f} GB"
                )
                logging.info(log_message)
                print(log_message)

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        avg_log_message = (
            f"Epoch {epoch} - Average Train Loss: {train_loss:.4f}, "
            f"Average Train Accuracy: {train_accuracy:.4f}"
        )
        logging.info(avg_log_message)
        print(avg_log_message)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.eval()
        val_loss, val_accuracy = 0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fc(out, y)
                val_loss += loss.item()
                val_accuracy += (y == torch.argmax(out, dim=-1)).float().mean().item()

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_save_path = str(
                WEIGHTS_DIR
                / f"best_model_wave_{height}_BS{batch_size}_FR{time_size}_val_acc_{best_val_accuracy:.4f}.pkl"
            )
            torch.save(model.state_dict(), model_save_path)

            test_log_message = f"New best model saved with accuracy: {best_val_accuracy:.4f}"
            logging.info(test_log_message)
            print(test_log_message)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == '__main__':
    train()
