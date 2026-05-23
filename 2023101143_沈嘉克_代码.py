from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
EPOCHS = 5
BATCH_SIZE = 64
VALIDATION_SIZE = 10_000
OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
CLASS_NAMES = [str(i) for i in range(10)]


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float


@dataclass
class ExperimentResult:
    experiment_type: str
    optimizer: str
    learning_rate: float
    test_loss: float
    test_accuracy: float
    model_path: str
    history: list[EpochMetrics]


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_check(log: Callable[[str], None]) -> torch.device:
    log("=== Environment Check ===")
    log(f"Python: {sys.version.split()[0]}")
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    tensor_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    tensor_b = torch.tensor([[2.0, 0.0], [1.0, 2.0]])
    log(f"Simple tensor operation result: {(tensor_a @ tensor_b).tolist()}")
    log("")
    return device


def get_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader, datasets.MNIST]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    full_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    train_size = len(full_train) - VALIDATION_SIZE
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, validation_dataset = random_split(
        full_train, [train_size, VALIDATION_SIZE], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, validation_loader, test_loader, full_train


def denormalize(image: torch.Tensor) -> np.ndarray:
    image = image.squeeze().detach().cpu().numpy()
    image = image * 0.3081 + 0.1307
    return np.clip(image, 0.0, 1.0)


def first_index_per_class(dataset: datasets.MNIST) -> list[int]:
    indices: dict[int, int] = {}
    for index in range(len(dataset)):
        _, label = dataset[index]
        label_value = int(label)
        if label_value not in indices:
            indices[label_value] = index
        if len(indices) == len(CLASS_NAMES):
            break
    return [indices[label] for label in range(len(CLASS_NAMES))]


def save_sample_images(dataset: datasets.MNIST, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax, index in zip(axes.ravel(), first_index_per_class(dataset)):
        image, label = dataset[index]
        ax.imshow(denormalize(image), cmap="gray")
        ax.set_title(f"True: {CLASS_NAMES[label]}")
        ax.axis("off")
    fig.suptitle("MNIST Training Samples", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    return run_one_epoch(model, loader, criterion, device, optimizer=None)


def experiment_slug(experiment_type: str, optimizer_name: str, learning_rate: float) -> str:
    lr_text = str(learning_rate).replace(".", "p")
    return f"{experiment_type}_{optimizer_name}_lr{lr_text}".lower().replace("+", "plus")


def train_experiment(
    experiment_type: str,
    optimizer_name: str,
    learning_rate: float,
    optimizer_factory: Callable[[nn.Module], optim.Optimizer],
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    log: Callable[[str], None],
) -> ExperimentResult:
    set_seed(SEED)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_factory(model)
    history: list[EpochMetrics] = []

    log(f"=== {experiment_type}: {optimizer_name} (lr={learning_rate}) ===")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_one_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        validation_loss, validation_accuracy = evaluate(model, validation_loader, criterion, device)
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
            )
        )
        log(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f}, train acc {train_accuracy:.4f} | "
            f"val loss {validation_loss:.4f}, val acc {validation_accuracy:.4f}"
        )

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    log(f"Test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.4f}")
    log("")

    slug = experiment_slug(experiment_type, optimizer_name, learning_rate)
    model_path = OUTPUT_DIR / f"model_{slug}.pt"
    torch.save(model.state_dict(), model_path)
    save_test_predictions(model, test_loader, device, OUTPUT_DIR / f"test_predictions_{slug}.png")

    return ExperimentResult(
        experiment_type=experiment_type,
        optimizer=optimizer_name,
        learning_rate=learning_rate,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        model_path=str(model_path),
        history=history,
    )


def save_test_predictions(model: nn.Module, loader: DataLoader, device: torch.device, output_path: Path) -> None:
    model.eval()
    selected_images: list[torch.Tensor] = []
    selected_labels: list[int] = []
    seen_labels: set[int] = set()
    for batch_images, batch_labels in loader:
        for image, label in zip(batch_images, batch_labels):
            label_value = int(label.item())
            if label_value not in seen_labels:
                seen_labels.add(label_value)
                selected_images.append(image)
                selected_labels.append(label_value)
            if len(seen_labels) == len(CLASS_NAMES):
                break
        if len(seen_labels) == len(CLASS_NAMES):
            break

    ordered = sorted(zip(selected_labels, selected_images), key=lambda item: item[0])
    labels = torch.tensor([label for label, _ in ordered])
    images = torch.stack([image for _, image in ordered])
    images = images.to(device)
    with torch.no_grad():
        predictions = model(images).argmax(dim=1).cpu()

    fig, axes = plt.subplots(2, 5, figsize=(11, 5.8))
    for ax, image, label, prediction in zip(axes.ravel(), images.cpu(), labels, predictions):
        ax.imshow(denormalize(image), cmap="gray")
        ax.set_title(f"True: {label.item()}  Pred: {prediction.item()}", fontsize=11)
        ax.axis("off")
    fig.suptitle("MNIST Test Predictions", fontsize=14)
    fig.subplots_adjust(top=0.86, hspace=0.45, wspace=0.08)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_group_curves(results: list[ExperimentResult], prefix: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for result in results:
        epochs = [item.epoch for item in result.history]
        train_loss = [item.train_loss for item in result.history]
        validation_loss = [item.validation_loss for item in result.history]
        label = f"{result.optimizer} lr={result.learning_rate:g}"
        ax.plot(epochs, train_loss, marker="o", linestyle="-", label=f"{label} train")
        ax.plot(epochs, validation_loss, marker="s", linestyle="--", label=f"{label} val")
    ax.set_title(f"{title} - Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{prefix}_validation_loss.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for result in results:
        epochs = [item.epoch for item in result.history]
        train_accuracy = [item.train_accuracy for item in result.history]
        validation_accuracy = [item.validation_accuracy for item in result.history]
        label = f"{result.optimizer} lr={result.learning_rate:g}"
        ax.plot(epochs, train_accuracy, marker="o", linestyle="-", label=f"{label} train")
        ax.plot(epochs, validation_accuracy, marker="s", linestyle="--", label=f"{label} val")
    ax.set_title(f"{title} - Training and Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{prefix}_validation_accuracy.png", dpi=180)
    plt.close(fig)


def plot_best_curves(result: ExperimentResult) -> None:
    epochs = [item.epoch for item in result.history]
    train_loss = [item.train_loss for item in result.history]
    validation_loss = [item.validation_loss for item in result.history]
    train_accuracy = [item.train_accuracy for item in result.history]
    validation_accuracy = [item.validation_accuracy for item in result.history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_loss, marker="o", label="Training Loss")
    ax.plot(epochs, validation_loss, marker="s", label="Validation Loss")
    ax.set_title("Best Model Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loss_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_accuracy, marker="o", label="Training Accuracy")
    ax.plot(epochs, validation_accuracy, marker="s", label="Validation Accuracy")
    ax.set_title("Best Model Accuracy Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=180)
    plt.close(fig)


def load_model(result: ExperimentResult, device: torch.device) -> SimpleCNN:
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(result.model_path, map_location=device))
    model.eval()
    return model


def save_conv_kernels(model: SimpleCNN, output_path: Path) -> None:
    kernels = model.features[0].weight.detach().cpu()
    fig, axes = plt.subplots(2, 4, figsize=(8, 4.2))
    for ax, kernel, index in zip(axes.ravel(), kernels[:8], range(8)):
        kernel_image = kernel.squeeze().numpy()
        ax.imshow(kernel_image, cmap="coolwarm")
        ax.set_title(f"Kernel {index}")
        ax.axis("off")
    fig.suptitle("First Convolution Layer Kernels", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_feature_maps(model: SimpleCNN, loader: DataLoader, device: torch.device, output_path: Path) -> None:
    image, label = next(iter(loader))
    single_image = image[:1].to(device)
    conv1 = model.features[0]
    relu = model.features[1]
    with torch.no_grad():
        maps = relu(conv1(single_image)).squeeze(0).cpu()

    fig, axes = plt.subplots(2, 4, figsize=(8, 4.2))
    for ax, fmap, index in zip(axes.ravel(), maps[:8], range(8)):
        ax.imshow(fmap.numpy(), cmap="viridis")
        ax.set_title(f"Map {index}")
        ax.axis("off")
    fig.suptitle(f"First Layer Feature Maps (true label: {label[0].item()})", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def collect_predictions(
    model: SimpleCNN, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[tuple[torch.Tensor, int, int]]]:
    all_labels: list[int] = []
    all_predictions: list[int] = []
    errors: list[tuple[torch.Tensor, int, int]] = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1).cpu()
            for image, label, prediction in zip(images, labels, predictions):
                true_label = int(label.item())
                pred_label = int(prediction.item())
                all_labels.append(true_label)
                all_predictions.append(pred_label)
                if true_label != pred_label:
                    errors.append((image, true_label, pred_label))
    return np.array(all_labels), np.array(all_predictions), errors


def save_misclassified_samples(
    errors: list[tuple[torch.Tensor, int, int]], output_path: Path, limit: int = 8
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(9, 4.8))
    for ax, (image, true_label, pred_label) in zip(axes.ravel(), errors[:limit]):
        ax.imshow(denormalize(image), cmap="gray")
        ax.set_title(f"True: {true_label}  Pred: {pred_label}", fontsize=10)
        ax.axis("off")
    for ax in axes.ravel()[len(errors[:limit]) :]:
        ax.axis("off")
    fig.suptitle("Misclassified Test Samples", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, output_path: Path) -> np.ndarray:
    matrix = np.zeros((10, 10), dtype=int)
    for true_label, pred_label in zip(labels, predictions):
        matrix[true_label, pred_label] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Test Set Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for row in range(10):
        for col in range(10):
            value = matrix[row, col]
            color = "white" if value > matrix.max() * 0.55 else "black"
            ax.text(col, row, str(value), ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return matrix


def confusion_summary(matrix: np.ndarray) -> dict[str, object]:
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    true_label, pred_label = np.unravel_index(np.argmax(off_diagonal), off_diagonal.shape)
    return {
        "most_confused_true": int(true_label),
        "most_confused_predicted": int(pred_label),
        "most_confused_count": int(off_diagonal[true_label, pred_label]),
        "per_class_accuracy": [
            float(matrix[index, index] / matrix[index].sum()) if matrix[index].sum() else 0.0
            for index in range(10)
        ],
    }


def save_results_csv(results: list[ExperimentResult], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Experiment Type",
                "Optimizer",
                "Learning Rate",
                "Final Training Loss",
                "Final Validation Loss",
                "Final Training Accuracy",
                "Final Validation Accuracy",
                "Test Accuracy",
            ]
        )
        for result in results:
            final = result.history[-1]
            writer.writerow(
                [
                    result.experiment_type,
                    result.optimizer,
                    result.learning_rate,
                    f"{final.train_loss:.4f}",
                    f"{final.validation_loss:.4f}",
                    f"{final.train_accuracy:.4f}",
                    f"{final.validation_accuracy:.4f}",
                    f"{result.test_accuracy:.4f}",
                ]
            )


def save_metrics_json(
    optimizer_results: list[ExperimentResult],
    lr_results: list[ExperimentResult],
    best_result: ExperimentResult,
    device: torch.device,
    confusion: dict[str, object],
) -> None:
    all_results = optimizer_results + lr_results
    payload = {
        "device": str(device),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "validation_size": VALIDATION_SIZE,
        "best_model": {
            "experiment_type": best_result.experiment_type,
            "optimizer": best_result.optimizer,
            "learning_rate": best_result.learning_rate,
            "test_loss": best_result.test_loss,
            "test_accuracy": best_result.test_accuracy,
            "model_path": best_result.model_path,
        },
        "confusion_summary": confusion,
        "results": [
            {
                **{key: value for key, value in asdict(result).items() if key != "history"},
                "history": [asdict(item) for item in result.history],
            }
            for result in all_results
        ],
        "figures": {
            "samples": "outputs/sample_images.png",
            "optimizer_loss": "outputs/optimizer_validation_loss.png",
            "optimizer_accuracy": "outputs/optimizer_validation_accuracy.png",
            "learning_rate_loss": "outputs/learning_rate_validation_loss.png",
            "learning_rate_accuracy": "outputs/learning_rate_validation_accuracy.png",
            "best_loss": "outputs/loss_curve.png",
            "best_accuracy": "outputs/accuracy_curve.png",
            "kernels": "outputs/conv1_kernels.png",
            "feature_maps": "outputs/feature_maps.png",
            "misclassified": "outputs/misclassified_samples.png",
            "confusion_matrix": "outputs/confusion_matrix.png",
            "predictions": "outputs/test_predictions.png",
        },
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    set_seed(SEED)
    device = environment_check(log)
    train_loader, validation_loader, test_loader, full_train = get_dataloaders()
    save_sample_images(full_train, OUTPUT_DIR / "sample_images.png")

    optimizer_specs = [
        ("Optimizer", "SGD", 0.01, lambda model: optim.SGD(model.parameters(), lr=0.01)),
        (
            "Optimizer",
            "SGD+Momentum",
            0.01,
            lambda model: optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        ),
        ("Optimizer", "Adam", 0.01, lambda model: optim.Adam(model.parameters(), lr=0.01)),
    ]
    optimizer_results = [
        train_experiment(kind, name, lr, factory, train_loader, validation_loader, test_loader, device, log)
        for kind, name, lr, factory in optimizer_specs
    ]

    lr_specs = [
        ("LearningRate", "Adam", 0.1, lambda model: optim.Adam(model.parameters(), lr=0.1)),
        ("LearningRate", "Adam", 0.01, lambda model: optim.Adam(model.parameters(), lr=0.01)),
        ("LearningRate", "Adam", 0.001, lambda model: optim.Adam(model.parameters(), lr=0.001)),
    ]
    lr_results = [
        train_experiment(kind, name, lr, factory, train_loader, validation_loader, test_loader, device, log)
        for kind, name, lr, factory in lr_specs
    ]

    plot_group_curves(optimizer_results, "optimizer", "Optimizer Comparison")
    plot_group_curves(lr_results, "learning_rate", "Adam Learning Rate Comparison")

    best_result = max(optimizer_results + lr_results, key=lambda item: item.test_accuracy)
    best_model = load_model(best_result, device)
    plot_best_curves(best_result)
    shutil.copyfile(
        OUTPUT_DIR
        / f"test_predictions_{experiment_slug(best_result.experiment_type, best_result.optimizer, best_result.learning_rate)}.png",
        OUTPUT_DIR / "test_predictions.png",
    )
    save_conv_kernels(best_model, OUTPUT_DIR / "conv1_kernels.png")
    save_feature_maps(best_model, test_loader, device, OUTPUT_DIR / "feature_maps.png")

    labels, predictions, errors = collect_predictions(best_model, test_loader, device)
    save_misclassified_samples(errors, OUTPUT_DIR / "misclassified_samples.png")
    matrix = save_confusion_matrix(labels, predictions, OUTPUT_DIR / "confusion_matrix.png")
    confusion = confusion_summary(matrix)

    save_results_csv(optimizer_results + lr_results, OUTPUT_DIR / "experiment_results.csv")
    save_metrics_json(optimizer_results, lr_results, best_result, device, confusion)
    (OUTPUT_DIR / "training_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
