import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import re
import io
from transformers import NougatProcessor, VisionEncoderDecoderModel

# Load the Nougat LaTeX model and processor
processor = NougatProcessor.from_pretrained("Norm/nougat-latex-base")
model = VisionEncoderDecoderModel.from_pretrained("Norm/nougat-latex-base")

# Function to extract text and equations using Nougat model
def extract_text_and_equations_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Extract regular text
        regular_text = page.get_text()
        
        # Convert the page to an image for equation extraction by the Nougat model
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Extract equations using Nougat model
        inputs = processor(images=img, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=1024)
        latex_output = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Combine regular text and LaTeX output
        combined_text = f"{regular_text}\n${latex_output}$\n"
        full_text += combined_text
    
    return full_text

# Function to extract images from PDF
def extract_images_from_pdf(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    image_count = 0
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))
            image_path = os.path.join(output_folder, f"image_{page_num+1}_{img_index+1}.{image_ext}")
            image.save(image_path)
            image_count += 1
    return image_count

# Function to save text to a file
def save_text_to_file(text, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)

# Main function to process all PDFs
def process_pdfs(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_output_folder = os.path.join(output_folder, "images")
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    for pdf_file in os.listdir(input_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, pdf_file)
            
            # Extract text and LaTeX equations using Nougat model
            text = extract_text_and_equations_from_pdf(pdf_path)
            
            # Extract images and save
            image_count = extract_images_from_pdf(pdf_path, image_output_folder)
            
            # Save text output
            text_file_path = os.path.join(output_folder, f"{os.path.splitext(pdf_file)[0]}.txt")
            save_text_to_file(text, text_file_path)
            print(f"Processed {pdf_file}: {image_count} images extracted")

# Set input and output folders
input_folder = "./test_pdfs"
output_folder = "./results"

process_pdfs(input_folder, output_folder)
