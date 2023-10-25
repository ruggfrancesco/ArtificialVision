import cv2
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing
import torch.nn.functional as F
import torchvision.transforms as transforms

torch.manual_seed(42)
torch.set_grad_enabled(True)
torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy("file_system")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

### Classes
class_labels = [
    "bottle-blue",
    "bottle-blue-full",
    "bottle-blue5l",
    "bottle-blue5l-full",
    "bottle-dark",
    "bottle-dark-full",
    "bottle-green",
    "bottle-green-full",
    "bottle-milk",
    "bottle-milk-full",
    "bottle-multicolor",
    "bottle-multicolorv-full",
    "bottle-oil",
    "bottle-oil-full",
    "bottle-transp",
    "bottle-transp-full",
    "bottle-yogurt",
    "canister",
    "cans",
    "detergent-box",
    "detergent-color",
    "detergent-transparent",
    "detergent-white",
    "glass-dark",
    "glass-green",
    "glass-transp",
    "juice-cardboard",
    "milk-cardboard",
]
num_classes = len(class_labels)
print(f"Number of classes: {num_classes}")


### Dataset params
num_workers = 8

img_width = 128
img_height = 128

mean_channel_1 = 0.337052
mean_channel_2 = 0.344904
mean_channel_3 = 0.351085
std_channel_1 = 0.180601
std_channel_2 = 0.176718
std_channel_3 = 0.183093


# Best model parameters
class params:
    batch_size = 16
    conv_units = 32
    dropout_rate = 0.0
    learning_rate = 0.0001
    weight_decay = 0.01


base_transform = transforms.Compose([
    # Resize image
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    # Color normalization
    transforms.Normalize(
        mean=(mean_channel_1, mean_channel_2, mean_channel_3),
        std=(std_channel_1, std_channel_2, std_channel_3),
    ),
])


# 2. Load Model
class CNN(nn.Module):
    def __init__(
        self,
        conv_units=None,
        learning_rate=None,
        weight_decay=None,
        dropout_rate=None,
        batch_size=None,
    ):
        super().__init__()
        self.name = f"CNN_{img_width}x{img_height}_{batch_size}bs_{learning_rate}lr_{weight_decay}wd_{conv_units}conv_{dropout_rate}dr"

        units = conv_units
        self.features_conv = nn.Sequential(
            nn.Conv2d(3, units, kernel_size=3, padding=1),
            nn.BatchNorm2d(units),
            nn.ReLU(),
            nn.Conv2d(units, units * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(units * 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(units * 2, units * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(units * 4),
            nn.ReLU(),
            nn.Conv2d(units * 4, units * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(units * 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(units * 4, units * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(units * 8),
            nn.ReLU(),
            nn.Conv2d(units * 8, units * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(units * 8),
            nn.ReLU(),
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_units * 8 * (img_width // 8) * (img_height // 8), 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

        self.gradients = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)

    def forward(self, x, register_hook=False):
        x = self.features_conv(x)
        if register_hook:
            h = x.register_hook(self.activations_hook)
        x = self.max_pool(x)
        x = self.classifier(x)
        return x


# 4. Predict video
# Open video and create an OpenCV window for displaying frames


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
   h_max = max(im.shape[0] for im in im_list)
   im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_max / im.shape[0]), h_max), interpolation=interpolation) for im in im_list]
   return cv2.hconcat(im_list_resize)

def evaluate_gradcam(frame):
    # Convert frame to PIL Image
    image = Image.fromarray(frame).convert("RGB")

    # Convert PIL Image to tensor and apply any necessary transformations
    image = base_transform(image).unsqueeze(0).to(device)

    pred = model(image, register_hook=True)
    labels_pred = pred.topk(5)
    pred.sum().backward()

    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.get_activations(image).detach()

    for i in range(64):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap_act = torch.mean(activations, dim=1).squeeze()
    heatmap_act = np.maximum(heatmap_act.cpu(), 0)
    heatmap_act /= torch.max(heatmap_act)

    heatmap = cv2.resize(heatmap_act.numpy(), (frame.shape[1], frame.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.5 + frame
    superimposed_img = np.uint8(superimposed_img)

    return frame, heatmap_act, superimposed_img, labels_pred


if __name__ == "__main__":
    # video_path = "/home/franci/Dropbox/university/VisioneArtificiale/project/paper/video/20230709_164309.mp4"
    # video_path = "/home/franci/Dropbox/university/VisioneArtificiale/project/video_test.mp4"

    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to the video file")
    args = parser.parse_args()
    video_path = args.video_path

    ### Load model
    model = CNN(
        conv_units=params.conv_units,
        learning_rate=params.learning_rate,
        weight_decay=params.weight_decay,
        dropout_rate=params.dropout_rate,
        batch_size=params.batch_size,
    )

    model_path = f"models/HT/CNN_128x128_16bs_0.0001lr_0.01wd_32conv_0.0dr.pt"
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Video fps: {fps}")

    frames = []
    while cap.isOpened():
        # read frames
        ret, frame = cap.read()

        if not ret:
            continue
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # evaluate grad-cam
        image_rgb, heatmap_act, superimposed_img, labels_pred = evaluate_gradcam(frame)

        # get predictions
        label_pred = labels_pred[1][0][0]
        pred_classes = list(zip(labels_pred[1][0].detach().cpu().numpy(), F.softmax(labels_pred[0], dim=1).squeeze().tolist()))
        pred_classes = [(class_labels[c]+":", np.round(s, 3)) for c,s in pred_classes]
        print(f"Frame labels_pred: {pred_classes}")

        time.sleep(1 / fps)

        # display prediction label and score on frames
        y0, dy, = 128, 48
        for i, (label, score) in enumerate(pred_classes):
            y = y0 + i*dy
            if i:
                cv2.putText(frame, f"{label:30} {score:5.3f}", (50, y), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2, cv2.LINE_AA, False)
            else:
                cv2.putText(frame, f"{label:20} {score:5.3f}", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)

        # display grad-cam
        heatmap_act_normalized = heatmap_act.squeeze().cpu().numpy()
        heatmap_act_normalized = (heatmap_act_normalized - heatmap_act_normalized.min()) / (heatmap_act_normalized.max() - heatmap_act_normalized.min())
        heatmap_act_scaled = (heatmap_act_normalized * 255).astype(np.uint8)
        heatmap_colormap = cv2.applyColorMap(heatmap_act_scaled, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colormap, cv2.COLOR_BGR2RGB)

        # merge and display original frame, grad-cam and superimposed image
        composite_img = hconcat_resize_min([frame, heatmap_rgb, superimposed_img])
        cv2.imshow("Image_Prediction_GradCam", composite_img)

    cap.release()
    cv2.destroyAllWindows()
