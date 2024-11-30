import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from tkinter import Tk, filedialog
from utils import gaussian_blur

# Load the pre-trained DeepLab model for segmentation
def load_model():
    print("Loading DeepLabV3 model with MobileNetV3...")
    from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
    weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', weights=weights)
    model.eval()
    return model

# Preprocess input frames for the DeepLab model
def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(frame_rgb)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    return input_tensor.unsqueeze(0)

# Perform segmentation on the frame to extract the mask
def segment_frame(model, input_batch, device):
    input_batch = input_batch.to(device)
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    return output.argmax(0).byte().cpu().numpy()

# Smooth the mask edges using Gaussian blur
def smooth_mask(mask, kernel_size=21, sigma=10):
    mask_float = mask.astype(np.float32)
    blurred = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), sigma)
    return np.clip(blurred, 0, 1)

# Open a file dialog to select a custom background image
def select_background_image():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Background Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if file_path:
        return cv2.imread(file_path)
    return None

# Initialize the color picker sliders for background customization
def initialize_color_picker():
    cv2.namedWindow('Background Color')
    cv2.createTrackbar('R', 'Background Color', 0, 255, lambda x: None)
    cv2.createTrackbar('G', 'Background Color', 0, 255, lambda x: None)
    cv2.createTrackbar('B', 'Background Color', 0, 255, lambda x: None)

# Retrieve the current background color from the sliders
def get_bg_color():
    r = cv2.getTrackbarPos('R', 'Background Color')
    g = cv2.getTrackbarPos('G', 'Background Color')
    b = cv2.getTrackbarPos('B', 'Background Color')
    return (b, g, r)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model()
    model.to(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    # Display available options to the user
    print("Commands:")
    print(" 1 - Segmentation overlay (person highlighted, original background)")
    print(" 2 - Person on a custom solid color background with smooth edges")
    print(" 3 - Person on a Gaussian blurred background with smooth edges")
    print(" 4 - Person on a custom background (select image)")
    print(" Q - Quit")

    selected_option = '1'
    custom_background = None
    initialize_color_picker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (256, 256))
        input_batch = preprocess_frame(resized_frame)
        model_predictions = segment_frame(model, input_batch, device)

        person_mask = (model_predictions == 15).astype(np.uint8)
        person_mask_resized = cv2.resize(person_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        smoothed_mask = smooth_mask(person_mask_resized, kernel_size=21, sigma=10)

        # Handle different options for background modification
        if selected_option == '1':
            mask_rgb = np.zeros_like(frame)
            mask_rgb[person_mask_resized == 1] = [0, 128, 255]
            output_frame = cv2.addWeighted(frame, 0.7, mask_rgb, 0.3, 0)

        elif selected_option == '2':
            bg_color = get_bg_color()
            solid_background = np.zeros_like(frame, dtype=np.uint8)
            solid_background[:, :] = bg_color
            output_frame = solid_background * (1 - smoothed_mask[..., None])
            output_frame += frame * smoothed_mask[..., None]

        elif selected_option == '3':
            blurred_background = gaussian_blur(frame[..., 0], kernel_size=15, sigma=10)
            blurred_background = cv2.merge([blurred_background] * 3)
            output_frame = blurred_background * (1 - smoothed_mask[..., None])
            output_frame += frame * smoothed_mask[..., None]

        elif selected_option == '4':
            if custom_background is not None:
                resized_background = cv2.resize(custom_background, (frame.shape[1], frame.shape[0]))
                output_frame = resized_background * (1 - smoothed_mask[..., None])
                output_frame += frame * smoothed_mask[..., None]
            else:
                output_frame = frame

        else:
            output_frame = frame

        cv2.imshow('Live Feed', np.uint8(output_frame))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('1'):
            selected_option = '1'
        elif key == ord('2'):
            selected_option = '2'
        elif key == ord('3'):
            selected_option = '3'
        elif key == ord('4'):
            selected_option = '4'
            custom_background = select_background_image()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
