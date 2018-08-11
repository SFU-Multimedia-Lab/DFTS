import numpy as np
from PIL import Image
import os
from .utils import pTasks
from tqdm import tqdm, trange
from bs4 import BeautifulSoup
from keras.preprocessing import image

class CFDataGenerator(object):
    """Generates preprocessed data in batches of given size.
    """
    def __init__(self, testData, **params):
        """
        # Arguments
            testData: directory location of the images
            params  : dictionary containing pre processing parameters
        """
        super(DataGenerator, self).__init__()
        self.testData    = testData
        self.reshapeDims = params['reshapeDims']
        self.batch_size  = params['batch_size']
        self.normalize   = params['normalize']
        self.n_classes   = params['n_classes']
        self.batch_index = 0
        self.runThrough  = False

    def getNextBatch(self):
        """Cycles through the data one batch_size at a time

        # Returns
            Current batch of preprocessed images
        """
        currentTestData = []
        if self.batch_index>=len(self.testData):
            #assume that batch_size<number of test images
            self.batch_index = 0
            currentTestData  = self.preprocess(self.testData[self.batch_index:self.batch_index+self.batch_size])
        elif self.batch_index + self.batch_size >= len(self.testData) and self.batch_index<len(self.testData):
            currentTestData  = self.preprocess(self.testData[self.batch_index:])
            self.batch_index = len(self.testData)
        else:
            currentTestData  = self.preprocess(self.testData[self.batch_index:self.batch_index+self.batch_size])
        if self.batch_index==len(self.testData):
            self.runThrough = True
        self.batch_index      += self.batch_size
        return currentTestData

    def preprocess(self, pdata):
        """Perform generic preprocessing on the images

        # Arguments
            pdata: batch of data to be preprocessed

        # Returns
            Array containing the labels and the preprocessed data.
        """
        labels = []
        data   = []
        for i in pdata:
            d = pTasks(i, self.reshapeDims, self.normalize)
            labels.append(int(i[0]))
            data.append(d)
        return (np.array(labels), np.array(data))


class ODDataGenerator(object):
    """docstring for ODDataGenerator"""
    def __init__(self, testDir, reshapeDims, batch_size, classes):
        super(ODDataGenerator, self).__init__()
        self.images      = testDir['images'] #each is a list
        self.imageSet    = testDir['testNames']
        self.annoDirs    = testDir['annotations']
        self.reshapeDims = reshapeDims
        self.batch_size  = batch_size
        self.classes     = classes
        self.batch_index = 0
        self.runThrough  = False
        self.parseData()

    def parseData(self):
        self.filenames = []
        self.imagesIds = []
        self.labels    = []

        for imdir, imfset, anndir in zip(self.images, self.imageSet, self.annoDirs):
            with open(imfset) as f:
                imagesIds = [line.strip() for line in f]
                self.imagesIds+=imagesIds

            it = tqdm(imagesIds, desc="Processing image set '{}'".format(os.path.basename(imfset)), file=sys.stdout)

            for imageId in it:
                filename = '{}'.format(imageId) + '.jpg'
                self.filenames.append(os.path.join(imdir, filename))

                with open(os.path.join(anndir, imageId+".xml")) as f:
                    soup = BeautifulSoup(f, 'xml')
                boxes = []
                objects = soup.find_all('object')

                for obj in objects:
                    class_name = obj.find('name', recursive=False).text
                    class_id = self.classes.index(class_name)
                    pose = obj.find('pose', recursive=False).text
                    bndbox = obj.find('bndbox', recursive=False)
                    xmin = int(bndbox.xmin.text)
                    ymin = int(bndbox.ymin.text)
                    xmax = int(bndbox.xmax.text)
                    ymax = int(bndbox.ymax.text)

                    item_dict = {'image_name': filename,
                                 'image_id': imageId,
                                 'class_name': class_name,
                                 'class_id': class_id,
                                 'pose': pose,
                                 'xmin': xmin,
                                 'ymin': ymin,
                                 'xmax': xmax,
                                 'ymax': ymax}
                    box = []
                    labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax')
                    for item in self.labels_output_format:
                        box.append(item_dict[item])
                    boxes = [item_dict['image_id']]
                    boxes.append(box)
                self.labels.append(boxes)

    def getNextBatch(self):
        currentTestData = []
        if self.batch_index>=len(self.filenames):
            self.batch_index = 0
            currentTestData = [self.preprocess(i, l, self.reshapeDims) for i, l in 
                               zip(self.filenames[self.batch_index:self.batch_index+self.batch_size], 
                               self.labels[self.batch_index:self.batch_index+self.batch_size])]
        elif self.batch_index + self.batch_size >= len(self.filenames) and self.batch_index<len(self.filenames):
            currentTestData = [self.preprocess(i, l, self.reshapeDims) for i, l in 
                               zip(self.filenames[self.batch_index:], self.labels[self.batch_index:])]
        else:
            currentTestData = [[self.preprocess(i, l, self.reshapeDims) for i, l in 
                               zip(self.filenames[self.batch_index:self.batch_index+self.batch_size], 
                               self.labels[self.batch_index:self.batch_index+self.batch_size])]]
        if self.batch_index==len(self.filenames):
            self.runThrough = True
        self.batch_index    += self.batch_size
        images = np.array([item[-1] for item in currentTestData])
        labels = np.array([item[1] for item in currentTestData])
        imageIds = np.array([item[0] for item in currentTestData])
        currentTestData = ((imageIds, labels), images)
        return currentTestData

    def preprocess(self, testImage, label, reshapeDims):
        I = image.load_img(testImage)
        imgH = I.size[0]
        imgW = I.size[1]
        I = I.resize(reshapeDims)
        I = image.img_to_array(I)
        label = labelReshape(label[1], imgH, imgW, reshapeDims)
        imageId = label[0]
        return (imageId, label, np.array(I))

    def labelReshape(label, imgH, imgW, reshapeDims):
        rH = reshapeDims[0]
        rW = reshapeDims[1]
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        for i in range(len(labels)):
            labels[i][xmin] = (labels[i][xmin]*rH)/imgH
            labels[i][xmax] = (labels[i][xmax]*rH)/imgH
            labels[i][ymin] = (labels[i][ymin]*rW)/imgW
            labels[i][ymax] = (labels[i][ymax]*rW)/imgW
        return labels

        
