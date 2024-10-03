import numpy as np
import cv2
import os
import imageio

save_path = './'
save_name = 'xxx.mp4'

img_array=[]

for filename in [f'./{str(i)}.jpg' for i in range(25)]:
    img = cv2.imread(filename)
    b,g,r = cv2.split(img)			
    img_new = cv2.merge([r,g,b])	
    if img is None:
        print(filename + " is error!")
        continue
    img_array.append(img_new)


size = (len(img_array[0]),len(img_array[0]))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
filename = os.path.join(save_path,save_name)

imageio.mimwrite(filename, img_array, fps=25, output_params=['-vf', 'fps=25'])
print('end!')