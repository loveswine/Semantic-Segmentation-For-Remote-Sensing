from PIL import Image
import numpy as np

# palette = [0, 200, 0,
# #            150, 250, 0,
# #            150, 200, 150,
# #            200, 0, 200,
# #            150, 0, 250,
# #            150, 150, 250,
# #            250, 200, 0,
# #            200, 200, 0,
# #            200, 0, 0,
# #            250, 0, 150,
# #            200, 150, 150,
# #            250, 150, 150,
# #            0, 0, 200,
# #            0, 150, 200,
# #            0, 200, 250,
# #            0, 0, 0]

palette = [0, 0, 0,
           83, 255, 160,
            125, 81, 54,
            238, 182, 35,
            0, 193, 255]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


# 将grey mask转化为彩色mask
def colorize_mask(mask):
    mask_color = Image.fromarray(mask.astype(np.uint8)).convert('P')
    mask_color.putpalette(palette)
    return mask_color

if __name__ == '__main__':
    lab=Image.open('./image/test/label/3-8bits.png').convert('P')
    lab.putpalette(palette)
    lab.save('./image/test/labelrgb/3-8bits.png')
    
    
    
