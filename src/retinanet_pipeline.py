"""
retinanet.py: Implementation of RetinaNet 
"""

import torch
from torchvision.models.detection import retinanet_resnet50_fpn

# Get model
def get_model():
    model = retinanet_resnet50_fpn(weights="DEFAULT")
    return model

# Run
def run():
    model = get_model()
    print("RetinaNet is found and ready")
    print(type(model))

# Main
if __name__ == "__main__":
    run()
