from sklearn.model_selection import train_test_split
import os

def preprocess(root_dir):
    if not os.path.exists(root_dir):
        print('The dataSet does not exist!')
    else:
        with open('./dataloaders/test.txt','w') as ftest:
            for label in os.listdir(root_dir): #每一个标签
                files_path = os.path.join(root_dir, label)
                img_files = [name for name in os.listdir(files_path)] #遍历每一个标签下的文件
                # train, test_img = train_test_split(img_files, test_size=0.1, random_state=42) #划分验证集和训练集
                for img in img_files:
                    ftest.write(os.path.join(files_path, img) + ' ' + label + '\n')
        ftest.close()
        print('Preprocessing finished.')
if __name__ == "__main__":
    root_path = 'D:/workplace/chongqing/GestureRecognition/DataSet/testData'
    preprocess(root_path)