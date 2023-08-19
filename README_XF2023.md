1.预训练模型的位置在models/pre.pt,如需更换预训练模型请将你的预训练模型放到这里，并修改名称为pre.pt

2.数据增广脚本在trick/LabelExpansion.py下，请将需要增广的数据集(包括.jpg和.txt)放到origin_data下，增广后的数据将保存到aug_data下,执行split_txtandjpg.py可以把图片和标签分开

3.在做数据增广之前请查看"trick/修改标签说明.doc"将原始数据中的误标数据修正

4.ultralytics/cfg/default.yaml是配置文件所在处

5.执行train.py将开始训练，模型将保存到runs/detec/train/weights下