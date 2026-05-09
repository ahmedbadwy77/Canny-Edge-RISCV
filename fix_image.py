from PIL import Image

try:
    # 1. Open the disguised image and convert to Black & White ('L')
    img = Image.open("fake_raw.jpg").convert('L')
    
    # 2. Force it to be exactly 256x256 pixels
    img = img.resize((256, 256))
    
    # 3. Extract the raw pixel bytes and save them
    with open("input.raw", "wb") as f:
        f.write(img.tobytes())
        
    print("Success! A true input.raw byte file has been created.")
except Exception as e:
    print("Error:", e)
