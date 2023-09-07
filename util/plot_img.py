from PIL import Image,ImageDraw
import numpy
import time


def draw_pil_img(img,targets):
    print(img.size)
    drawObject = ImageDraw.Draw(img)
    for line in  targets['boxes']:
        line = line.cpu().numpy()
        x1,y1,x2,y2 = line
        drawObject.rectangle([x1,y1,x2,y2], fill=None, outline=None, width=3)

    img.show()
    time.sleep()




