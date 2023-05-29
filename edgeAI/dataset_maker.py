#imports
import os
import cv2
import random

random.seed(2000) #set a constant seed for all runs
crop_iteration_size=5
noise_amplitude=3
dimensions=[420,900]
dataset=[]
 
# set the path/directory
folder_dir = "edgeAI/inputs"
images_names=os.listdir(folder_dir) #get dir list of filenames
for image_name in images_names:
    # check if the image_name ends with PNG just to be sure
    if (image_name.endswith(".PNG")):
        print(image_name)
        img = cv2.imread(folder_dir+'/'+image_name)
        flip_variations=[img,]
        for i in range(-1,2):
            flip_variations.append(cv2.flip(img, i))
        crop_variations=[]
        for image in flip_variations:
            for iteration in range(crop_iteration_size):
                h,w,c=image.shape
                temp=image[random.randint(0,noise_amplitude):h-random.randint(0,noise_amplitude), random.randint(0,noise_amplitude):w-random.randint(0,noise_amplitude)].copy()
                crop_variations.append(temp)
        dataset.extend(crop_variations)
    
for i,image in enumerate(dataset):
    #cv2.imshow('image',image)
    #cv2.waitKey(0)
    cv2.imwrite("edgeAI/dataset/image{}.PNG".format(i+1),image)



