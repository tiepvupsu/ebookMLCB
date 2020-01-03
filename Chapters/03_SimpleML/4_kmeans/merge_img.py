import sys
from PIL import Image

images = map(Image.open, ['output_12_2.png', 'output_12_3.png'])

widths, heights = zip(*(i.size for i in images))


total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.save('k_15_20.png')