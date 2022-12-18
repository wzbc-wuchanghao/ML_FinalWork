import os
import json

data_path = '/home/daslab/nfs/wch/homework/ml/fabric_defect/'

annotation_path = 'annotations/'

train_2007 = '2007_train_defect.txt'
val_2007 = '2007_val_defect.txt'

crop_trgt = 'crop_trgt/'

with open(data_path+annotation_path+'train.json') as f, open(train_2007, 'w') as ft:
    train_json = json.loads(''.join(f.readlines()))
    images_list = [img['id'] for img in train_json['images']]
    lines = []
    for t in train_json['annotations']:
        
        idx = images_list.index(t['image_id'])
        if(idx>=0):
            file_name = train_json['images'][idx]['file_name']
            file_name = data_path + crop_trgt + file_name

            line = file_name + ' ' + str(t['bbox'][0]) + ',' + str(t['bbox'][1])+ ',' + str(t['bbox'][0] + t['bbox'][2])+ ',' + str(t['bbox'][1] + t['bbox'][3]) +',1\n'
            lines.append(line)
    ft.writelines(lines)

with open(data_path+annotation_path+'val.json') as f, open(val_2007, 'w') as ft:
    train_json = json.loads(''.join(f.readlines()))
    images_list = [img['id'] for img in train_json['images']]
    lines = []
    for t in train_json['annotations']:
        
        idx = images_list.index(t['image_id'])
        if(idx>=0):
            file_name = train_json['images'][idx]['file_name']
            file_name = data_path + crop_trgt + file_name

            line = file_name + ' ' + str(t['bbox'][0]) + ',' + str(t['bbox'][1])+ ',' + str(t['bbox'][0] + t['bbox'][2])+ ',' + str(t['bbox'][1] + t['bbox'][3]) +',1\n'
            lines.append(line)
    ft.writelines(lines)