# 把已经打好的标签按照类别分类
# 目的是找到含有数量较少的类别，进行单独增广

import os
import glob
import shutil
txt=glob.glob(r'C:\Users\ps\Desktop\2\demo' +'\\*.txt') # 图片和标签存放路径
obj_path=r'C:\Users\ps\Desktop\demo'   #保存结果路径

if not os.path.exists(obj_path):
    os.makedirs(obj_path)
label0=13  #移动的标签
for t in txt:
    with open(t,'r+') as f:
        lines=f.readlines()
        print(lines)
        f.close()
    label=[int(i.split(' ')[0]) for i in lines]
    if label0 in label:
        txt_name=os.path.basename(t)
        jpg=os.path.join(os.path.dirname(t),txt_name.split('.')[0] +'.jpg')
        shutil.move(t,obj_path)
        shutil.move(jpg, obj_path)