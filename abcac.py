import requests

def perform_ocr(image_path):
    url = "https://ocr.sanskritdictionary.com/recognise"
    
    # Text payload parameters
    data_payload = {
        "lang": "san",
        "service": "google"
    }
    
    try:
        # Open the image file in binary read mode
        with open(image_path, 'rb') as image_file:
            # The 'files' parameter handles the multipart/form-data encoding automatically
            files = {
                "image": (image_path, image_file, "image/jpeg") # Adjust content-type if using png
            }
            
            # Make the POST request
            print("Sending request...")
            response = requests.post(url, data=data_payload, files=files)
            
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()
            
            # Parse the JSON response
            result = response.json()
            
            print("\n--- OCR Result ---")
            print(result.get("text", "No text found in response."))
            
    except FileNotFoundError:
        print(f"Error: Could not find the image file at '{image_path}'")
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed: {e}")
    except ValueError:
        print("Failed to parse JSON response. Raw response:")
        print(response.text)

# --- How to use the function ---
if __name__ == "__main__":
    # Replace 'your_image.jpg' with the actual path to the image you want to scan
    image_file_path = "aaa.png" 
    
    perform_ocr(image_file_path)