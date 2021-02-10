#融合多个数据集
from sklearn.model_selection import train_test_split
import os

def preprocess(root_dir1, root_dir2,root_dir3):
    if not os.path.exists(root_dir1):
        print('The dataSet does not exist!')
    else:
        with open('./dataloaders/train.txt','w') as ftrain , open('./dataloaders/valid.txt','w') as fvalid:
            for label in os.listdir(root_dir1): #每一个标签
                files_path = os.path.join(root_dir1, label)
                files_path2 = os.path.join(root_dir2, label)
                files_path3 = os.path.join(root_dir3, label)
                # img_files = [name for name in os.listdir(files_path)] #遍历每一个标签下的文件
                img_files = []
                for name in os.listdir(files_path):
                    img_files.append(os.path.join(files_path, name))
                for name in os.listdir(files_path2):
                    img_files.append(os.path.join(files_path2, name))
                for name in os.listdir(files_path3):
                    img_files.append(os.path.join(files_path3, name))
                train, valid= train_test_split(img_files, test_size=0.2, random_state=42) #划分验证集和训练集
                for tra in train:
                    ftrain.write(tra + ' ' + label + '\n')
                for val in valid:
                    fvalid.write(val + ' ' + label + '\n')
        ftrain.close()
        fvalid.close()
        print('Preprocessing finished.')
if __name__ == "__main__":
    root_path = 'D:/workplace/chongqing/GestureRecognition/DataSet/bothDataSet'
    root_paht2 = 'D:/workplace/chongqing/GestureRecognition/DataSet/others'
    root_paht3 = 'D:/workplace/chongqing/GestureRecognition/DataSet/others2/other2'
    preprocess(root_path, root_paht2,root_paht3)