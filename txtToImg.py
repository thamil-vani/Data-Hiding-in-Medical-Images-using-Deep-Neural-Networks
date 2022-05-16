# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 19:46:14 2021

@author: Thamil Vani
"""


from PIL import Image, ImageDraw
 
img = Image.new('RGB', (64,64), color = (0, 0, 0))
 
d = ImageDraw.Draw(img)
d.text((10,10), "Hello \nWorld", fill=(255,255,0))
 
img.save('pil_text.png')