import os
from PIL import Image

dir = 'test_results'

l = 28
size = [8, 8]
grid = Image.new('RGB', size=(size[0]*l, size[1]*l))

count = 0

# Create image stack
for entry in os.scandir(dir):
    if entry.name[-10:] == 'attmap.png':
        with open(entry, 'rb') as file:
            img = Image.open(file)
            grid.paste(img, box=(count%size[0]*l, count//size[1]*l))
        count += 1

grid.save(os.path.join(dir, 'grid.png'))  
