from PIL import Image
import os

# Make sure the file exists
if not os.path.exists("output.raw"):
    print("Error: output.raw not found!")
    exit()

# Read the raw bytes from your C++ output
with open("output.raw", "rb") as f:
    data = f.read()

# Convert bytes back into a 256x256 grayscale image
img = Image.frombytes('L', (256, 256), data)
img.save("final_edges.png")
print("Success! Saved as final_edges.png")
