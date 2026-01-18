import requests

# URL of your running FastAPI
url = "http://127.0.0.1:8000/predict"

# Open an image you want to test
image_path = "S_test.jpg"  # replace with your image file
files = {"file": open(image_path, "rb")}

# Send request
response = requests.post(url, files=files)

# Print the result
print(response.json())
