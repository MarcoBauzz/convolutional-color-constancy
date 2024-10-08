import os
import torch
from PIL import Image
import csv
import numpy as np

class ColorCheckerDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root='.', gt='GT.txt', sets=['all'], extension=None, transforms=None):
        
        self.root = root
        self.transforms = transforms

        # Read Ground truth file with file names, illuminants, and folds
        # (TODO: use proper column names instead of counting on correct order)
        self.file_list = []
        rgb_list = []
        fold_list = []
        with open(gt) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for (line_count,row) in enumerate(csv_reader):
                if line_count > 0 or row[0]!='File':
                    self.file_list.append(row[0])
                    rgb_list.append([float(r) for r in row[1:4]])
                    fold_list.append(int(row[4]))
        self.rgb_array = np.array(rgb_list, dtype='single')
        self.fold_array = np.array(fold_list)
        self.len = line_count

        # Subsets to keep
        if 'all' not in sets:
            to_keep = np.full((line_count), False)

            if 'test1' in sets or '1' in sets or 1 in sets or 'fold1' in sets or 'test' in sets:
                to_keep = np.logical_or(to_keep, self.fold_array == 1)
            if 'test2' in sets or '2' in sets or 2 in sets or 'fold2' in sets or 'train' in sets:
                to_keep = np.logical_or(to_keep, self.fold_array == 2)
            if 'test3' in sets or '3' in sets or 3 in sets or 'fold3' in sets or 'valid' in sets:
                to_keep = np.logical_or(to_keep, self.fold_array == 3)

            if 'train1' in sets:
                to_keep = np.logical_or(to_keep, self.fold_array != 1)
            if 'train2' in sets:
                to_keep = np.logical_or(to_keep, self.fold_array != 2)
            if 'train3' in sets:
                to_keep = np.logical_or(to_keep, self.fold_array != 3)

            self.file_list = [f for (f, tk) in zip(self.file_list, to_keep) if tk]
            self.rgb_array = self.rgb_array[to_keep,:]
            self.fold_array = self.fold_array[to_keep]
            self.len = len(self.file_list)

        # Replace extension if necessary
        # TODO Works only if there IS an extension to start with
        if extension is not None:
            if extension[0] == '.':
                extension = extension[1:]
            self.file_list = [f[:f.rfind('.')]+'.'+extension for f in self.file_list]

        print('Total files in GT: {:d}. Requested: {:d}'.format(line_count, self.len))


    def __getitem__(self, index):

        img = Image.open(os.path.join(self.root, self.file_list[index]))#.astype('float32')
        if self.transforms is not None:
            img = self.transforms(img)

        rgb = self.rgb_array[index,:]

        return (img, rgb)

    def __len__(self):
        return self.len


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from matplotlib import pyplot as plt

    dataset = ColorCheckerDataset(root='/data/Datasets/ColorConstancy/ColorChecker/Hemrit/masked_long800/',
                                  gt='/data/Datasets/ColorConstancy/ColorChecker/GT/GT_HemritRec.txt',
                                  sets=[1],
                                  extension='.png',
                                  transforms=transforms.Compose([transforms.ToTensor()])
                                  )
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

    for batch_idx, (img, rgb) in enumerate(dataloader):
        ipdb.set_trace()
        print(batch_idx)
