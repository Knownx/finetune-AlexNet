'''''''''''''''''''''''''''''''''''''''''''''''''''''
   create time: 2018.03.14 Wed. 23h45m16s
   author: Chuanfeng Liu
   e-mail: microlj@126.com
   github: https://github.com/Knownx
'''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
	Implementation of extract the file paths and labels into a text
'''
import os
def listImg(path):
    train = open('D:\\Workspace\\ML\\Proj\\OpenSource\\finetune_alexnet\\train\\train.txt','w')
    file_path = path
    for file in os.listdir(path):
        classes = file.split('.')[0]
        if classes == 'cat':
            label = '0'
        elif classes == 'dog':
            label = '1'
        else:
            raise NameError('Class name isn\'t correct!')
        train.write(file_path + '\\' + file + ' ' + label + '\n')

#listImg('D:\\Workspace\\ML\\Proj\\OpenSource\\finetune_alexnet\\train\\train')

def listTestImg(path):
    test = open('D:\\Workspace\\ML\\Proj\\OpenSource\\finetune_alexnet\\test\\test.txt','w')
    file_path = path
    for file in os.listdir(path):
        test.write(file_path + '\\' + file + '\n')

#listTestImg('D:\\Workspace\\ML\\Proj\\OpenSource\\finetune_alexnet\\test\\test')