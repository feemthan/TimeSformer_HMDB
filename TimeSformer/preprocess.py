# Import necessary packages
import os
import subprocess
from PIL import Image

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
from tqdm import tqdm

root_directory = 'HMDB_simp'
train_directory = 'video_outputs'
new_size = (224, 224)

file_paths = []
for root, dirs, files in os.walk(root_directory):
    for file in files:
        if file.endswith('.jpg'):
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

for image_path in tqdm(file_paths, desc="Resizing images", unit="image"):
    image = Image.open(image_path)
    resized_image = image.resize(new_size)
    resized_image.save(image_path)

if not os.path.exists(train_directory):
    os.makedirs(train_directory)
else:
    pass

# call ffmpeg to generate videos via subprocess
root_dir = root_directory
for classes in os.listdir(root_directory):
    for folders in os.listdir(os.path.join(root_directory, classes)):
        output_dir = train_directory + '/' + classes
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            pass
        subprocess.call(['ffmpeg', '-y', '-r', '30', '-pattern_type', 'sequence','-i', 'HMDB_simp/'+classes+'/'+folders+'/%04d.jpg', output_dir +'/'+folders+'.mp4'])

test_dir = 'test_data/'
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
else:
    pass

subprocess.run(['cp', '-r', train_directory, test_dir], check=True)

# Train, test and val dataset preparation
current_dir = os.getcwd()
class_path = []
for dir_path, _, files in os.walk(train_directory):
    for file in files:
        if not file.endswith('.csv'):
            paths = os.path.join(current_dir, dir_path, file)
            class_name = dir_path.split('/')[-1]
            class_path.append([f"{paths} {class_name}"])

# convert to dataframe to be run by train_test_split
dataset = pd.DataFrame(class_path)
dataset.to_csv('class_path.csv', index=False, header=False)

file_path = dataset.iloc[:, 0].str.split(' ', expand=True)[0]
class_name = dataset.iloc[:, 0].str.split(' ', expand=True)[1]

label_encoder = LabelEncoder()
class_name = label_encoder.fit_transform(class_name)

# map the labels to numbers
label_to_num_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
label_to_num_map = {k: int(v) for k, v in label_to_num_map.items()}

# build json file to remap in the future
with open('label_mapping.json', 'w') as file:
    json.dump(label_to_num_map, file)

# split multiple times for 80 10 10, put your student ID here
STUDENT_ID = 2538
X_train, X_temp, y_train, y_temp = train_test_split(file_path, class_name, test_size=0.2, random_state=STUDENT_ID)
X_test1, X_test2, y_test1, y_test2 = train_test_split(X_temp, y_temp, test_size=0.5, random_state=STUDENT_ID)

# convert to dataframe
train_df = pd.DataFrame({'filepath': X_train, 'class': y_train})
test1_df = pd.DataFrame({'filepath': X_test1, 'class': y_test1})
val_df = pd.DataFrame({'filepath': X_test2, 'class': y_test2})

# save to csv without index or header
train_df.to_csv('video_outputs/train.csv', index=False, header=False, sep=' ')
val_df.to_csv('video_outputs/val.csv', index=False, header=False, sep=' ')
test1_df.to_csv('video_outputs/test.csv', index=False, header=False, sep=' ')