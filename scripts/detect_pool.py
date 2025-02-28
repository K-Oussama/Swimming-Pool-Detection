#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import sys

def detect_pools(image_path, output_image_path, output_coordinates_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)
    orig_image = image.copy()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([80, 70, 100])
    upper_blue = np.array([130, 255, 255])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    edges = cv2.Canny(mask, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    min_area = 300
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            filtered_contours.append(contour)
    
    
    cv2.drawContours(orig_image, filtered_contours, -1, (255, 0, 0), 1)
    
    
    cv2.imwrite(output_image_path, orig_image)
    print(f"Output image saved as {output_image_path}")
    
    with open(output_coordinates_path, "w") as f:
        for cnt in filtered_contours:
            coords = cnt.reshape(-1, 2)
            for point in coords:
                f.write(f"{point[0]},{point[1]}\n")
            f.write("\n") 
    
    print(f"Coordinates saved in {output_coordinates_path}")

def main():
    parser = argparse.ArgumentParser(description="CLI script to detect swimming pools in aerial images using OpenCV.")
    parser.add_argument("--input", required=True, help="Path to the input aerial image.")
    parser.add_argument("--output", required=True, help="Path to save the output image with detected pool contours (e.g., output_image.jpg).")
    parser.add_argument("--coordinates", required=True, help="Path to save the output coordinates file (e.g., coordinates.txt).")
    
    args = parser.parse_args()
    detect_pools(args.input, args.output, args.coordinates)

if __name__ == "__main__":
    main()