from numpy import asarray
from PIL import Image
import numpy as np
import pandas as pd
import os

filepath = 'images'

def clean_image_data(filepath):

    if os.path.exists('cleaned_images') == False:
        os.mkdir('cleaned_images')

    dirs = os.listdir(filepath)
    final_size = 64
    for n, item in enumerate(dirs, 1):
        file_name = item.split('.')[0]
        im = Image.open(os.path.join(filepath, item))
        new_im = resize_image(final_size, im)
        new_im.save(f'cleaned_images/{file_name}_cleaned.jpg')


def resize_image(final_size, im: Image):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

def Create_arrays():
    path = "cleaned_images/"
    df = pd.read_csv('Image+Products.csv', lineterminator="\n")
    id_list = df['image_id'].to_list()
    img_df = pd.DataFrame()
    for id in id_list:
        im = Image.open(path + id + '_cleaned.jpg')
        img = np.mean(im, axis=2) # convert color image to gray
        im_array = asarray(img)
        img_df = img_df.append({"image": [im_array]})
    df_final = pd.merge(df, img_df)
    print(df_final)
    #df_final.to_pickle('products+arrays.pkl')

#clean_image_data(filepath)
#Create_arrays()