#!/usr/bin/env python
# coding: utf-8

# In[1]:


#coverting image from one formato another
from PIL import Image
im = Image.open('ing.png')
im.convert ('RGB').save('ing.jpg')
print('image converted sucess')


# In[2]:


im = Image.open('ing.png')
print(image.mode)


# In[ ]:


import cv2
from PIL import Image
doggo=Image.open('ing.png')
doggo.imshow(doggo);
import matplotlib.pyplot as plt
fig, ax=plt.subplots(1, 3, figsize=(12,4), sharey =True)
ax[0].imshow(doggo[:,:,0], cmap= "Reds")
ax[0].set_title('Red')
ax[1].imshow(doggo[:,:,1], cmap= "Greens")
ax[1].set_title('Green')
ax[2].imshow(doggo[:,:,2], cmap= "Blues")
ax[2].set_title('Blue');


# In[1]:


from PIL import Image, ImageDraw
im=Image.open('ing.png')
draw=ImageDraw.Draw(im)
draw.ellipse((125,125,200,250),fill(255,255,255,120))
del draw
im.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
img = np.array(Image.open('ing.png'))
img_R, img_G, img_B = img.copy(), img.copy(), img.copy()
img_R[:, :, (1, 2)] = 0
img_G[:, :, (0, 2)] = 0
img_B[:, :, (0, 1)] = 0
img_rgb = np.concatenate((img_R,img_G,img_B), axis=1)
plt.figure(figsize=(15, 15))
plt.imshow(img_rgb)


# In[ ]:


#slicing

fig, ax=plt.subplots(1,3,figsize=(6,4), sharey = True)
ax[0].imshow(doggo[:, 0:130])
ax[0].set_title('First split')

ax[1].imshow(doggo[:, 130:260])
ax[1].set_title('First split')

ax[2].imshow(doggo[:, 260:390])
ax[2].set_title('First split');


# In[1]:


#blending
from PIL import Image
img1=Image.open('star.png')
img2=Image.open('b.png')
alpha1=Image.blend(img1,img2,alpha=.4)
alpha2=Image.blend(img1,img2,alpha=.2)
alpha1.show()
alpha2.show()


# In[ ]:


#histogram
im=Image.open('ing.png')
pl=im.histogram()
plt.bar(range(256),pl[:256], color='r', alpha=0.5)
plt.bar(range(256),pl[256:2*256], color='g', alpha=0.4)
plt.bar(range(256),pl[2*256:], color='b', alpha=0.3)
plt.show()


# In[ ]:


ch_r,ch_g,ch_b=im.split()
plt.figure(figsize=(18,6))
plt.subplot(1,3,1); plt.imshow(ch_r, cmap=plt.cm.Reds);plt.axis('off')
plt.subplot(1,3,2); plt.imshow(ch_g, cmap=plt.cm.Greens);plt.axis('off')
plt.subplot(1,3,3); plt.imshow(ch_b, cmap=plt.cm.Blues);plt.axis('off')     
plt.tight_layout()
plt.show()


# In[ ]:


from PIL import Image,ImageDraw,ImageFont
img =Image.open('ing.png')
d1=ImageDraw.Draw(img)
myFont =ImageFont.truetype('arial.ttf',40)
d1.text((500,500),"Sample text", font=myFont, fill=(25,8,0))
img.show()


# In[ ]:


#croped image
cropped = im.crop((1,2,300,300))

cropped.show()


# In[ ]:


converting 
from skimage.io import imshow, imread
from PIL import Image
doggo=imread('ing.png')
imshow(doggo);
import matplotlib.pyplot as plt
fig, ax=plt.subplots(1, 3, figsize=(12,4), sharey =True)
ax[0].imshow(doggo[:,:,0], cmap= "Reds")
ax[0].set_title('Red')
ax[1].imshow(doggo[:,:,1], cmap= "Greens")
ax[1].set_title('Green')
ax[2].imshow(doggo[:,:,2], cmap= "Blues")
ax[2].set_title('Blue');


# In[2]:


import PIL

#negative transformation function
def neg_trans(img):

  #get width and height of image
  width,height=img.size

  #traverse through pixels
  for x in range(width):
    for y in range(height):

      pixel_color=img.getpixel((x,y))

      #if image is RGB, subtract individual RGB values
      if type(pixel_color) == tuple: 

        #s=(L-1)-r
        red_pixel=256-1-pixel_color[0]
        green_pixel=256-1-pixel_color[1]
        blue_pixel=256-1-pixel_color[2]

        #replace the pixel 
        img.putpixel((x,y),(red_pixel,green_pixel,blue_pixel))
      
      #if image is greyscale, subtract pixel intensity
      else:

        #s=(L-1)-r
        pixel_color=256-1-pixel_color 

        #replace the pixel
        img.putpixel((x,y),pixel_color)
  return img


# In[7]:


#importing required libraries
import numpy as np
from pylab import imshow, show


# In[8]:


from PIL import Image,ImageStat
img=Image.open('ing2.png')
stat1=ImageStat.Stat(img)
print(stat1.mean)
  


# In[9]:


print(stat1.median)


# In[10]:


print(stat1.stddev)


# In[ ]:




