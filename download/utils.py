from urllib.request import urlretrieve
import os

def downloadModel(url):
    print('Model download started...')
    fileName = url.split('/')[-1]
    filePath = f'../kerasModels/{fileName}'

    if not os.path.exists(os.path.dirname(filePath)):
        try:
            os.makedirs(os.path.dirname(fileName))
        except OSError as exc:
            if exc.errno != errno.EXIST:
                raise

    urlretrieve(url, fileName)
    print('Model download completed')
    return fileName
