import numpy as np
import os
import time
import datetime
# from tqdm import tqdm
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
sys.path.append(os.getcwd())
from segmentmap import segment_map_generator
from matplotlib import pyplot as plt


# Global variables
root_dir = r'D:/Soumi/License plate detection/LP-night/'
# test_train_sep_dir = os.path.join(citycam_dir, 'C:/Users/CVPR/source/repos/Detection/CityCam/train_test_separation')
annotation_path = os.path.join(root_dir, 'Annotation_Test.txt')
# downtown_test_path = os.path.join(test_train_sep_dir, 'C:/Users/CVPR/source/repos/Detection/CityCam/train_test_separation/Downtown_Test.txt')
# parkway_train_path = os.path.join(test_train_sep_dir, 'C:/Users/CVPR/source/repos/Detection/CityCam/train_test_separation/Parkway_Train.txt')
# parkway_test_path = os.path.join(test_train_sep_dir, 'C:/Users/CVPR/source/repos/Detection/CityCam/train_test_separation/Parkway_Test.txt')

# Global variables




def _parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    print(xml_path)
    width = round(int(root.find('.//size/width').text)/2)
    height = round(int(root.find('.//size/height').text)/2)
    
    # assert height == 240
    # assert width == 352
    
    objects = root.findall('.//object')
    center_points = []
    coordinates=[]
    for object in objects:
        bnd_box = object.find('bndbox')
        x_max = round(int(bnd_box.find('xmax').text)/2)
        x_min = round(int(bnd_box.find('xmin').text)/2)
        y_max = round(int(bnd_box.find('ymax').text)/2)
        y_min = round(int(bnd_box.find('ymin').text)/2)
        coordinates.append((x_max,x_min,y_max,y_min))

        if x_min <= 0:
            x_min = 0
        if x_max >= width:
            x_max = width
        if y_min <= 0:
            y_min = 0
        if y_max >= height:
            y_max = height
        
        x_center = (x_max + x_min) // 2
        y_center = (y_max + y_min) // 2

        center_points.append((y_center, x_center))
        
        
    return (height, width), center_points,coordinates

def time_stamp() -> str:
    ts = time.time()
    time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return time_stamp

def _make_density_map(frame_dir):
    # img_shape (height, width)
    frame_list = sorted(os.listdir(frame_dir))
    # frame_dir1=os.path.join('numpyfile')
    for frame in frame_list:
        frame_id, frame_format = frame.split('.')
        if frame_format == 'xml':
            if os.path.exists(os.path.join(frame_dir, frame_id + '_hm' + '.npy')):
                continue

            img_shape, points, coordinates = _parse_xml(os.path.join(frame_dir, frame))
            try:
                img_shape, points, coordinates = _parse_xml(os.path.join(frame_dir, frame))
                density_map = segment_map_generator(img_shape, points,coordinates)
            except AssertionError as e:
                print(frame_dir)
                raise e
            except Exception as e:
                print(frame_dir)
                print(e)
                raise e
            else:
                np.save(os.path.join(frame_dir, frame_id + '_hm'), density_map)


def make_density_map(target_dir):
    print('Processing: ' + time_stamp() + ' - ' + target_dir)
    camera_dir = os.path.join(root_dir, target_dir.split('-')[0])
    frame_dir = os.path.join(camera_dir, target_dir)
    _make_density_map(frame_dir)


if __name__ == "__main__":
    train_list = []
    with open(annotation_path) as f:
        train_list.extend(f.readlines())
    
    # with open(parkway_train_path) as f:
    #     train_list.extend(f.readlines()[:15])

    # test_list = []
    # with open(downtown_test_path) as f:
    #     test_list.extend(f.readlines()[:20])

    # with open(parkway_test_path) as f:
    #     test_list.extend(f.readlines()[:10])

    train_list = [sample.strip() for sample in train_list]
    # test_list = [sample.strip() for sample in test_list]

    print('Computing training set density maps...')

    with ProcessPoolExecutor(max_workers=None) as executor:
        results = executor.map(make_density_map, train_list)
    
    tuple(results)

