try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse
    
    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter
    
    
    print("PIL imports:")
    from PIL import Image
    
    print("matplotlib imports:")
    
    import matplotlib.pyplot as plt

except ImportError:
    print("Need to fix the installation")
    raise

print("All imports okay. Yay!")


from Controller.run_attention import get_lights_coordinates
from random import randint


def walks_inside_the_folders(folder_name, ending_img):
    list_to_fill = []
    
    for file_name in os.listdir(folder_name):
        
        if file_name.endswith(ending_img):
            list_to_fill += ["".join((folder_name, "/", file_name))]

        else:
            current_path = "".join((folder_name, "/", file_name))
            if os.path.isdir(current_path):
                list_to_fill += walks_inside_the_folders(current_path, ending_img)
        
    return list_to_fill


def get_img_and_label(folder_name):
    default_base = 'Cityscapes'
    list_images = walks_inside_the_folders(default_base + '/leftImg8bit/' + folder_name, '_leftImg8bit.png')
    list_labels = walks_inside_the_folders(default_base + '/gtFine/' + folder_name, '_gtFine_labelIds.png')

    for image,labels in zip(list_images,list_labels):
        yield image, labels


def get_padding_img(img: np.array):
    temp = np.zeros((1104, 2128, 3), dtype=np.uint8)
    temp[40:1064, 40:2088] = img
    
    return temp


def get_crop_img(img, coordinate):
    x, y = coordinate

    return img[x - 40:x + 41, y - 40: y + 41]


def random_coordinates(axis_x, axis_y):    
    while axis_x:
        rand_index = randint(0, len(axis_x) - 1)
        x = axis_x[rand_index]
        y = axis_y[rand_index] 
        axis_x.pop(rand_index)
        axis_y.pop(rand_index)

        yield x, y
        

def get_crops_imgs_and_labels(src_img, label_img):
    axis_x, axis_y = get_lights_coordinates(src_img.convert('L'))
    
    amount_match = 5
    count_traffic_lights = 0
    count_non_traffic_lights = 0
    crops_img_tl = []
    labels_tl = []
    crops_img_non_tl = []
    labels_non_tl = []

    src = np.asarray(src_img)
    label = np.asarray(label_img)
    padding_img = get_padding_img(src)

    for x, y in random_coordinates(list(axis_x), list(axis_y)):
        if label[x, y] == 19 and count_traffic_lights < amount_match:
            count_traffic_lights += 1
            crop_img = get_crop_img(padding_img, (x + 40, y + 40))
            crops_img_tl += [crop_img]
            labels_tl += [1]

        if label[x, y] != 19 and count_non_traffic_lights < amount_match:
            count_non_traffic_lights += 1
            crop_img = get_crop_img(padding_img, (x + 40, y + 40))
            crops_img_non_tl += [crop_img]
            labels_non_tl += [0]

        if count_non_traffic_lights == amount_match and count_traffic_lights == amount_match:
            break

    min_amount = min(count_traffic_lights, count_non_traffic_lights)
    crops_img = crops_img_tl[:min_amount] + crops_img_non_tl[:min_amount]
    labels = labels_tl[:min_amount] + labels_non_tl[:min_amount]

    return crops_img, labels


def build_dataset():
    for src_img_name, label_img_name in get_img_and_label("train"):
        src_img = Image.open(src_img_name)
        label_img = Image.open(label_img_name)
        crops_imgs, labels = get_crops_imgs_and_labels(src_img, label_img)
        
        if crops_imgs:
            with open('Data_dir_random_updated/train/data.bin', "ab") as data_file:
                for img in crops_imgs:
                    np.array(img, dtype=np.uint8).tofile(data_file)
                
            with open('Data_dir_random_updated/train/labels.bin', "ab") as labels_file:
                for label in labels:
                    labels_file.write((label).to_bytes(1, byteorder='big', signed=False))
    
    for src_img_name, label_img_name in get_img_and_label("val"):
        src_img = Image.open(src_img_name)
        label_img = Image.open(label_img_name)
        crops_imgs, labels = get_crops_imgs_and_labels(src_img, label_img)
        
        if crops_imgs:
            with open('Data_dir_random_updated/val/data.bin', "ab") as data_file:
                for img in crops_imgs:
                    np.array(img, dtype=np.uint8).tofile(data_file)
                
            with open('Data_dir_random_updated/val/labels.bin', "ab") as labels_file:
                for label in labels:
                    labels_file.write((label).to_bytes(1, byteorder='big', signed=False))


def display_crop_img_and_label(img_file, label_file, index):   
    fpo = np.memmap(img_file, dtype=np.uint8, mode='r', offset=index * 19683)
    fpo = fpo[:19683]
    img = fpo.reshape(81, 81 ,3)
    Image.fromarray(img.astype('uint8'), 'RGB').show()
    
    fpo = np.memmap(label_file, mode='r', offset=index, dtype=np.uint8)
    label = fpo[0]
    
    print('label :' ,label)


if __name__ == "__main__":
    build_dataset()
    # display_crop_img_and_label('Data_dir/train/data.bin', 'Data_dir/train/labels.bin' ,17)
    # display_crop_img_and_label('Data_dir/train/data.bin', 'Data_dir/train/labels.bin' ,38)
    # display_crop_img_and_label('Data_dir/val/data.bin', 'Data_dir/val/labels.bin' ,0)
    # display_crop_img_and_label('Data_dir/val/data.bin', 'Data_dir/val/labels.bin' ,3)