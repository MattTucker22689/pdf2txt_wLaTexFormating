import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import re
import cv2
import numpy as np
from pix2tex.cli import LatexOCR

# Set up directories
input_dir = "./test_pdfs"
output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)

# Initialize LatexOCR
latex_ocr = LatexOCR()

def process_pdf(pdf_path):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_dir = os.path.join(output_dir, pdf_name)
    os.makedirs(pdf_output_dir, exist_ok=True)
    
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    
    text_content = ""
    image_count = 0
    
    for i, image in enumerate(images):
        # Convert PIL Image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Perform OCR
        text = pytesseract.image_to_string(image)
        
        # Detect equations
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 20:  # Adjust these thresholds as needed
                roi = opencv_image[y:y+h, x:x+w]
                equation = latex_ocr(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)))
                text = text.replace(pytesseract.image_to_string(roi), f"$${equation}$$")
        
        # Detect and save images, graphs, and tables
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 100:  # Adjust these thresholds as needed
                roi = opencv_image[y:y+h, x:x+w]
                image_count += 1
                image_filename = f"image_{image_count}.png"
                cv2.imwrite(os.path.join(pdf_output_dir, image_filename), roi)
                text = text.replace(pytesseract.image_to_string(roi), f"[Image {image_count}]")
        
        text_content += text + "\n\n"
    
    # Write text content to file
    with open(os.path.join(output_dir, f"{pdf_name}.txt"), "w", encoding="utf-8") as f:
        f.write(text_content)

# Process all PDFs in the input directory
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith('.pdf'):
            pdf_path = os.path.join(root, file)
            process_pdf(pdf_path)

print("Processing complete. Results are in the './results/' directory.")
