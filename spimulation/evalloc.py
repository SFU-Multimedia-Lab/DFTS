from modelTasks.eval import *

def evalAllocater(task, metrics, reshapeDims, classes):
	if task==0:
		return CFeval(metrics, reshapeDims, classes)
	elif task==1:
		return ODeval(metrics, reshapeDims, classes)