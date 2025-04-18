from PIL import Image

def resize_with_padding(image_path, output_path, target_size=(1080, 720)):
    image = Image.open(image_path).convert("RGB")
    image.thumbnail(target_size, Image.LANCZOS)
    new_image = Image.new("RGB", target_size, (255, 255, 255))
    x_offset = (target_size[0] - image.size[0]) // 2
    y_offset = (target_size[1] - image.size[1]) // 2
    new_image.paste(image, (x_offset, y_offset))
    new_image.save(output_path)
    print(f"Saved resized image: {output_path}")

