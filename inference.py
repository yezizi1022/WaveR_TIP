import os
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from train import ClipsDataset  # Import your dataset class
from wave import ViViT
from thop import profile  # For FLOPs and parameter profiling
from torchvision import transforms
import time

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = Path(__file__).resolve().parent
DATA_ROOT = os.environ.get("WAVER_DATA_ROOT", str(ROOT_DIR / "data" / "LaparoClipsP2"))
LOG_DIR = ROOT_DIR / "logs"
WEIGHTS_DIR = ROOT_DIR / "weights"
OUTPUT_DIR = ROOT_DIR / "output"
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set up logging
log_file_name = str(LOG_DIR / "wave.log")
logging.basicConfig(
    filename=log_file_name,
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Helper functions
def plot_confusion_matrix(cm, class_names, vmax=250):

    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues,
               vmin=0, vmax=vmax)
    plt.colorbar()
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    thresh = vmax / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}",
                 ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "wave_confusion_matrix.jpg"))
    plt.show()

def calculate_gflops_and_parameters(model, input_size):
    model.eval()  # Set the model to evaluation mode
    dummy_input = torch.randn(1, *input_size[1:], device=device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    gflops = flops / 1e9
    params_m = params / 1e6  # Convert parameters to millions
    return gflops, params_m

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

def calculate_topk5_accuracy(outputs, labels, k=5):
    topk_preds = outputs.topk(k, dim=1)[1]  # Get top-k predictions
    topk_correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
    topk_accuracy = topk_correct.any(dim=1).float().mean().item()
    return topk_accuracy

def calculate_topk1_accuracy(outputs, labels, k=1):
    topk_preds = outputs.topk(k, dim=1)[1]  # Get top-k predictions
    topk_correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
    topk_accuracy = topk_correct.any(dim=1).float().mean().item()
    return topk_accuracy

def test(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    top5_correct = 0
    top1_correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            # Accuracy calculation
            correct += (preds == y).sum().item()
            total += y.size(0)

            # Top-5 Accuracy calculation
            top5_correct += calculate_topk5_accuracy(outputs, y, k=5) * y.size(0)
            # Top-1 Accuracy calculation
            top1_correct += calculate_topk1_accuracy(outputs, y, k=1) * y.size(0)

    # Accuracy
    accuracy = correct / total
    top5_accuracy = top5_correct / total
    top1_accuracy = top1_correct / total
    log_message = f"Accuracy: {accuracy:.4f}, Top-5 Accuracy: {top5_accuracy:.4f}, Top-1 Accuracy: : {top1_accuracy:.4f} "
    logging.info(log_message)  # Save to log
    print(log_message)  # Print to console

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)

    # Precision, Recall, F1-Score
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Log all metrics
    metrics_log = (f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
                   f"F1-Score: {f1:.4f}")
    logging.info(metrics_log)
    print(metrics_log)

if __name__ == "__main__":
    # Test dataset and loader setup
    test_data_path = DATA_ROOT
    test_csv_file = os.path.join(DATA_ROOT, "test.csv")

    # Define the transform (same as validation in training)
    test_transform = transforms.Compose([transforms.ToTensor()])

    # Initialize dataset and loader
    test_dataset = ClipsDataset(test_data_path, test_csv_file, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model setup
    model_path = str(
        WEIGHTS_DIR / "best_model.pkl"
    )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {model_path}. "
            "Train a model with train.py first or place a .pkl checkpoint in the weights/ directory."
        )

    class_names = [
        "AbdominalEntry", "Excise", "HookCut", "Incise", "LocPanoView",
        "NeedleIn", "NeedleOut", "PanoView", "ScissorCut", "Suction", "UseClip"
    ]

    height = width = 512
    img_size = height  # Since height = width
    time_size = 16

    model = ViViT(height, 32, 11, time_size)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Compute GFLOPs and parameters
    input_size = next(iter(test_loader))[0].shape
    gflops, params_m = calculate_gflops_and_parameters(model, input_size)
    logging.info(f"GFLOPs: {gflops:.4f}, Parameters (M): {params_m:.2f}")
    print(f"GFLOPs: {gflops:.4f}, Parameters (M): {params_m:.2f}")

    # Compute inference speed
    inference_speed = calculate_inference_speed(model, test_loader)
    logging.info(f"Inference Speed (s/sample): {inference_speed:.4f}")
    print(f"Inference Speed (s/sample): {inference_speed:.4f}")

    # Run testing
    test(model, test_loader, class_names)
