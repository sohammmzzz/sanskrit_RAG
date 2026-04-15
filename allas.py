# pip install requests pdf2image python-docx

import requests
import time
import os
from pdf2image import convert_from_path

try:
    import docx
except ImportError:
    docx = None

# --- Configuration ---
INPUT_FILE = "input.pdf"            # Your PDF or DOCX file path
OUTPUT_FOLDER = "ocr_output"        # Folder to save images/text
OCR_URL = "https://sanskritdictionary.com" # Endpoint
LANGUAGE = "san"                    # 'san' for Sanskrit

# Headers to mimic a real browser (prevents blocking)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Origin": "https://ocr.sanskritdictionary.com",
    "Referer": "https://ocr.sanskritdictionary.com/"
}

def process_pdf():
    print(f"Processing PDF: {INPUT_FILE}...")
    try:
        pages = convert_from_path(INPUT_FILE, dpi=300)
    except Exception as e:
        print(f"Error converting PDF: {e}")
        print("Ensure Poppler is installed and in your PATH.")
        return []

    full_text = []

    for i, page in enumerate(pages):
        image_filename = f"{OUTPUT_FOLDER}/page_{i+1}.jpg"
        text_filename = f"{OUTPUT_FOLDER}/page_{i+1}.txt"
        
        # Save image temporarily
        page.save(image_filename, "JPEG")
        
        print(f"Uploading Page {i+1}...", end=" ", flush=True)

        try:
            # Open the image in binary mode
            with open(image_filename, 'rb') as img_file:
                files = {'image': img_file}
                data = {'language': LANGUAGE}
                
                # Send POST request
                response = requests.post(OCR_URL, files=files, data=data, headers=HEADERS)
            
            if response.status_code == 200:
                # The site returns the raw text directly in the body
                extracted_text = response.text.strip()
                
                # Save individual page text
                with open(text_filename, "w", encoding="utf-8") as f:
                    f.write(extracted_text)
                
                full_text.append(f"--- Page {i+1} ---\n{extracted_text}\n")
                print("Success!")
            else:
                print(f"Failed [Status: {response.status_code}]")
        
        except Exception as e:
            print(f"Error: {e}")

        # Be polite to the server
        time.sleep(2)
        
    return full_text

def process_docx():
    print(f"Processing DOCX: {INPUT_FILE}...")
    if docx is None:
        print("Error: python-docx is not installed. Please install it using: pip install python-docx")
        return []
        
    try:
        doc = docx.Document(INPUT_FILE)
        full_text = []
        for i, para in enumerate(doc.paragraphs):
            if para.text.strip():
                full_text.append(para.text + "\n")
                
        extracted_text = "".join(full_text)
        
        text_filename = f"{OUTPUT_FOLDER}/docx_text.txt"
        with open(text_filename, "w", encoding="utf-8") as f:
            f.write(extracted_text)
            
        return [extracted_text]
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return []

def run_extraction():
    # 1. Create output directory
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File '{INPUT_FILE}' does not exist.")
        return

    ext = os.path.splitext(INPUT_FILE)[1].lower()
    full_text = []

    # 2. Extract text based on file extension
    if ext == '.pdf':
        full_text = process_pdf()
    elif ext in ['.docx', '.doc']:
        full_text = process_docx()
    else:
        print(f"Unsupported file format: {ext}. Please provide a .pdf or .docx file.")
        return

    # 3. Compile full text
    if full_text:
        final_output = os.path.join(OUTPUT_FOLDER, "complete_text.txt")
        with open(final_output, "w", encoding="utf-8") as f:
            f.writelines(full_text)
        print(f"\nDone! Complete text saved to: {final_output}")
    else:
        print("\nNo text could be extracted.")

if __name__ == "__main__":
    run_extraction()
