def absoluteFilePaths(directory):
    #push to utils
   dirList = os.listdir(directory)
   dirList = [os.path.join(directory, i) for i in dirList]
   dirList = [os.path.abspath(i) for i in dirList]
   return dirList
