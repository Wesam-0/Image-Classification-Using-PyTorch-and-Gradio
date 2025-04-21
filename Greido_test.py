import torch
import requests
from PIL import Image
import gradio as gr
from torchvision import transforms


# Download human-readable labels for ImageNet.
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")


def predict(inp):
 inp = transforms.ToTensor()(inp).unsqueeze(0)
 with torch.no_grad():
  prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
  confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
 return confidences

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),  # User uploads an image; it's automatically converted to a PIL Image
    outputs=gr.Label(num_top_classes=3),  # Display top 3 predicted labels with confidence scores
    examples=["/content/lion.jpg", "/content/cheetah.jpg"],  # Example images for quick testing
    title="Image Classification with ResNet-18",  # Title displayed at the top of the interface
    description="Upload an image to classify it using a pretrained ResNet-18 model from PyTorch."  # Short description for users
).launch()


