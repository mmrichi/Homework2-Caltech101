from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys

def group_indices_by_value(tuples):
    v = {}
    for index, (_, value) in enumerate(tuples):
        v.setdefault(value, set()).add(index)
    
    return v

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        '''

        classes, class_to_idx = self._find_classes(self.root, 'BACKGROUND_Google')
        imgs = self._make_split(self.root, class_to_idx) # [(path, class_index)]
        
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 files in subfolders"))

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.imgs = imgs
        self.targets = [s[1] for s in imgs]        
        self.indices_by_value = group_indices_by_value(self.imgs)

    def _find_classes(self, dir, class_to_filter):
        '''
        Args:
            dir: (str) directory
            class_to_filter: (str) class to be ignore in the dataset
        Returns:
            classes: list of directory names
            class_to_idx: {name: id_class}
        '''

        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.remove(class_to_filter)
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self, directory, class_to_idx):
        '''
        for files in **directory** (and sub-directories), matches their paths with their **class_index**
        only categories inside **class_to_idx** are matched in **instances**

        Returns: 
            instances: list of tuples (path, class_index)
        '''

        instances = []
        directory = os.path.expanduser(directory) # ... + 'Caltech101/101_ObjectCategories'
        
        for target_class in sorted(class_to_idx.keys()): # class_to_idx.key() == dir_name == category
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class) # ... + 'Caltech101/101_ObjectCategories' + '/[target_dir]'
            if not os.path.isdir(target_dir): # if exists and is dir 
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)): # root, dirs, files = os.walk() generates the file names in a directory tree
                for fname in sorted(fnames):
                    path = os.path.join(root, fname) # ... + 'Caltech101/101_ObjectCategories' + '/[target_dir]' + '/[fname]'
                    item = path, class_index
                    instances.append(item)
        return instances # [(path, class_index)]

    def _make_split(self, directory, class_to_idx):
        path = os.path.expanduser(directory) # ... + 'Caltech101/101_ObjectCategories'
        object_categories_dir = directory.split('/')[1]
        path = path[:-len(object_categories_dir)] # ... + 'Caltech101'

        fname = self.split + '.txt'
        split_path = path + fname

        instances = []
        with open(split_path, 'r') as split_file:
            for file in split_file.readlines():
                category = file.split('/')[0]
                if category != 'BACKGROUND_Google':
                    file_path = directory + '/' + file[:-1] # [:-1] strips out end of line
                    class_idx = class_to_idx[category]
                    item = file_path, class_idx
                    instances.append(item)
        return instances

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        path, label = self.imgs[index]
        image = pil_loader(path)

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        '''

        return len(self.imgs)
