import os
import random
from PIL import Image

# Path to malignant images
data_dir = 'data/processed/baseline/train/malignant'
output_path = 'malignant_grid_4x4.png'

def get_image_paths(directory):
    return [os.path.join(directory, fname) for fname in os.listdir(directory)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

def create_image_grid(image_paths, grid_size=(4, 4), image_size=(128, 128)):
    grid_w, grid_h = grid_size
    img_w, img_h = image_size
    grid_img = Image.new('RGB', (grid_w * img_w, grid_h * img_h))
    for idx, img_path in enumerate(image_paths):
        if idx >= grid_w * grid_h:
            break
        img = Image.open(img_path).convert('RGB').resize((img_w, img_h))
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid_img.paste(img, (x, y))
    return grid_img

def main():
    image_paths = get_image_paths(data_dir)
    if len(image_paths) < 16:
        raise ValueError(f'Not enough images in {data_dir} to create a 4x4 grid.')
    selected_paths = random.sample(image_paths, 16)
    grid_img = create_image_grid(selected_paths)
    grid_img.save(output_path)
    print(f'Grid saved to {output_path}')

if __name__ == '__main__':
    main()
