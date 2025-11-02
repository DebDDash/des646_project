"""
Local Explainability Module (LIME)

This module uses LIME (Local Interpretable Model-Agnostic Explanations)
to generate human-understandable visual explanations for model predictions.

It can help reveal which image regions most strongly influenced a model’s decision —
useful for debugging dataset bias or verifying if the model is "looking" at the right features.

Dependencies:
    pip install lime scikit-image torch torchvision numpy matplotlib
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries


# -----------------------------
# 1️⃣ Preprocessing Utility
# -----------------------------
def get_preprocess_transform():
    """Return standard preprocessing for ImageNet-based models."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# -----------------------------
# 2️⃣ Model Wrapper for LIME
# -----------------------------
class TorchModelWrapper:
    """
    Wrap a PyTorch model so it can be used with LIME.
    The wrapper converts a NumPy image batch into a tensor, applies preprocessing,
    and returns probabilities for each class.
    """

    def __init__(self, model, device=None):
        self.model = model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocess = get_preprocess_transform()

    def predict(self, images):
        """
        LIME expects a batch of RGB images (NumPy arrays) of shape (H, W, 3).
        Returns: np.ndarray of softmax probabilities (N, num_classes)
        """
        tensors = []
        for img in images:
            img = Image.fromarray(np.uint8(img))
            tensors.append(self.preprocess(img))
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.nn.functional.softmax(outputs, dim=1)

        return probs.cpu().numpy()


# -----------------------------
# 3️⃣ Generate LIME Explanation
# -----------------------------
def explain_image_with_lime(model, image_path, class_names=None, top_label=None, num_samples=1000):
    """
    Generate and visualize a LIME explanation for an image.

    Args:
        model: pretrained classifier (PyTorch)
        image_path: str, path to input image
        class_names: optional list of class names for labeling
        top_label: class index to explain (default = model’s predicted class)
        num_samples: number of LIME perturbations
    Returns:
        explanation (lime_image.ImageExplanation)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = TorchModelWrapper(model, device=device)
    explainer = lime_image.LimeImageExplainer()

    # Load and preprocess image for display
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # Run LIME
    explanation = explainer.explain_instance(
        image=img_np,
        classifier_fn=wrapper.predict,
        top_labels=5,
        hide_color=0,
        num_samples=num_samples
    )

    # If not specified, use the model's top predicted label
    if top_label is None:
        top_label = explanation.top_labels[0]

    # Get mask for the positive region
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=8,
        hide_rest=False
    )

    # Visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title(f"LIME explanation — class: {class_names[top_label] if class_names else top_label}")
    plt.axis("off")
    plt.show()

    return explanation


# -----------------------------
# 4️⃣ Batch Mode (Optional)
# -----------------------------
def batch_explain(model, image_paths, class_names=None, save_dir="lime_outputs", limit=10):
    """
    Generate LIME explanations for a small batch of images (for visual dataset auditing).

    Args:
        model: trained model
        image_paths: list[str]
        class_names: optional class labels
        save_dir: directory to save explanations
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    for img_path in image_paths[:limit]:
        print(f"Explaining {img_path} ...")
        explanation = explain_image_with_lime(model, img_path, class_names=class_names)
        save_path = os.path.join(save_dir, os.path.basename(img_path).replace(".jpg", "_lime.png"))
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    print(f"Saved {min(limit, len(image_paths))} explanations to {save_dir}/")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    from torchvision import models

    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)

    # Test with one example image
    image_path = "sample_images/dog.jpg"
    explain_image_with_lime(model, image_path, class_names=["cat", "dog", "car", "plane"])
