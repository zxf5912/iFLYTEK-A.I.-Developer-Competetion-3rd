import os
import shutil

# 把jpg和txt分开
files = os.listdir('../aug_data')

for file in files:
    print(file)

    if file[-1] == 'g':
        shutil.move('../aug_data/'+file,'../aug_data/images/'+file)

    elif file[-1] == 't':
        shutil.move('../aug_data/'+file,'../aug_data/labels/'+file)

