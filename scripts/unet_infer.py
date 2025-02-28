#!/usr/bin/env python3
import argparse
import sys
import cv2
import numpy as np
import tensorflow as tf

def load_and_infer(model_path, input_image_path, output_image_path, output_coordinates_path):
    # Load the pre-trained U-Net model
    #model_path = "unet_model.h5"
    print("Loading model from:", model_path)
    model = tf.keras.models.load_model(model_path)
    
    # Load the input image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Unable to load image at {input_image_path}")
        sys.exit(1)
   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    # Resize image to the model's expected input size (adjust as needed; here assumed 256x256)
    target_size = (512, 512)
    resized_image = cv2.resize(image, target_size)
    input_data = resized_image.astype(np.float32) / 255.0  # Normalize to [0,1]
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    
    # Run model inference to get the predicted mask
    pred_mask = model.predict(input_data)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # Binarize the mask

    
    # Resize the predicted mask back to the original image size
    #mask_resized = cv2.resize(pred_mask, (image.shape[1], image.shape[0]))
    
    # Find contours on the predicted mask
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the detected contours on the original image (b;ue color)
    orig_image = image.copy()
    cv2.drawContours(orig_image, contours, -1, (0, 0, 255), 1)
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    
    # Save the output image with contours
    cv2.imwrite(output_image_path, orig_image)
    print(f"Output image saved as {output_image_path}")
    
    # Save the coordinates of the detected contours to a text file
    with open(output_coordinates_path, "w") as f:
        for cnt in contours:
            coords = cnt.reshape(-1, 2)  # Reshape to (x, y) format
            for point in coords:
                f.write(f"{point[0]},{point[1]}\n")
            f.write("\n")  # Separate contours
    print(f"Coordinates saved in {output_coordinates_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CLI script to load a pre-trained U-Net model and run inference on an aerial image to detect swimming pools."
    )
    parser.add_argument("--model", required=True, help="Path to the saved U-Net model (e.g., unet_model.h5)")
    parser.add_argument("--input", required=True, help="Path to the input aerial image")
    parser.add_argument("--output", required=True, help="Path to save the output image with detected contours (e.g., output_image.jpg)")
    parser.add_argument("--coordinates", required=True, help="Path to save the output coordinates (e.g., coordinates.txt)")
    
    args = parser.parse_args()
    load_and_infer(args.model,args.input, args.output, args.coordinates)

if __name__ == "__main__":
    main()






