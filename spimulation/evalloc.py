from modelTasks.eval import *

def evalAllocater(task, metrics, reshapeDims):
	if task==0:
		return CFeval(metrics, reshapeDims)
	elif task==1:
		return ODeval(metrics, reshapeDims)