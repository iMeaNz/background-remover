import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
    model.eval()
    return model

def preprocess_frame(frame):
    # Convert the frame (BGR to RGB) and prepare it as a PIL image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(frame_rgb)

    # Apply the required preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def segment_frame(model, input_batch, device):
    input_batch = input_batch.to(device)
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    return output_predictions

def get_mask(mask, colors):
    # Create an RGB version of the mask
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id in range(len(colors)):
        mask_rgb[mask == class_id] = colors[class_id]

    return mask_rgb

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = load_model()
    model.to(device)

    num_classes = 21
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(num_classes)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        resized_frame = cv2.resize(frame, (256, 256))

        input_batch = preprocess_frame(resized_frame)

        model_predictions = segment_frame(model, input_batch, device)

        mask = get_mask(model_predictions, colors)

        resized_frame = cv2.resize(resized_frame, (520, 520))
        mask = cv2.resize(mask, (520, 520))

        cv2.imshow('Original Frame', resized_frame)
        cv2.imshow('Segmentation Overlay', mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
