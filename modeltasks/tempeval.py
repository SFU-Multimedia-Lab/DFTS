import numpy as np
from collections import Counter
import sys

def calcmAp(labels, predictions, IOUThreshold, imageIds, n_classes):
	n_classes = 20

	groundTruths = []
	detections = predictions

	ret = []

	for i in range(len(imageIds)):
		imageBoxes = labels[i]
		for j in range(len(imageBoxes)):
			boxes = imageBoxes[j]

			b = list(boxes)
			b.insert(0, imageIds[i])
			b.insert(2, 1)
			# print(b)
			groundTruths.append(b)
	# print(groundTruths)

	for c in range(1, n_classes+1):
		dects = detections[c]
		#pred format: image_id, confidence, xmin, ymin, xmax, ymax
		#gt format: image_id, 'class_id', conf,  'xmin', 'ymin', 'xmax', 'ymax'
		
		gts = []
		[gts.append(g) for g in groundTruths if g[1]==c]
		npos = len(gts)

		dects = sorted(dects, key=lambda conf: conf[1], reverse=True)
		TP = np.zeros(len(dects))
		FP = np.zeros(len(dects))

		det = Counter([cc[0] for cc in gts])
		# print(c)
		# print(det)
		for key, val in det.items():
			det[key] = np.zeros(val)

		for d in range(len(dects)):
			gt = [gt for gt in gts if gt[0]==dects[d][0]]
			iouMax = sys.float_info.min

			for j in range(len(gt)):
				iou = evalIOU(dects[d][2:], gt[j][3:])
				if iou>iouMax:
					iouMax = iou
					jmax = j
			if iouMax>=IOUThreshold:
				if det[dects[d][0]][jmax] == 0:
					TP[d] = 1
				det[dects[d][0]][jmax] = 1
			else:
				FP[d] = 1
		acc_FP = np.cumsum(FP)
		acc_TP = np.cumsum(TP)
		# if npos==0:
		# 	ap = 0
		# else:
		rec = acc_TP/npos
		prec = np.divide(acc_TP,(acc_FP+acc_TP))
		[ap, mpre, mrec, ii] = CalculateAveragePrecision(rec, prec)
		# print(ap)
		ret.append(ap)
	tot = len(ret)
	print(ret)
	return np.nansum(ret)/tot

def evalIOU(boxes1, boxes2):
	boxes1 = np.array(boxes1)
	boxes2 = np.array(boxes2)

	if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
	if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

	xmin = 0
	ymin = 1
	xmax = 2
	ymax = 3

	intersection_areas = intersection_area_(boxes1, boxes2)
	boxes1_areas = (boxes1[:, xmax] - boxes1[:, xmin] + 1) * (boxes1[:, ymax] - boxes1[:, ymin] + 1)
	boxes2_areas = (boxes2[:, xmax] - boxes2[:, xmin] + 1) * (boxes2[:, ymax] - boxes2[:, ymin] + 1)

	union_areas = boxes1_areas + boxes2_areas - intersection_areas

	return intersection_areas / union_areas

def intersection_area_(boxes1, boxes2):
	xmin = 0
	ymin = 1
	xmax = 2
	ymax = 3

	min_xy = np.maximum(boxes1[:,[xmin,ymin]], boxes2[:,[xmin,ymin]])
	max_xy = np.minimum(boxes1[:,[xmax,ymax]], boxes2[:,[xmax,ymax]])

	# Compute the side lengths of the intersection rectangles.
	side_lengths = np.maximum(0, max_xy - min_xy + 1)

	return side_lengths[:,0] * side_lengths[:,1]


def CalculateAveragePrecision(rec, prec):
	mrec = []
	mrec.append(0)
	[mrec.append(e) for e in rec]
	mrec.append(1)
	mpre = []
	mpre.append(0)
	[mpre.append(e) for e in prec]
	mpre.append(0)
	for i in range(len(mpre)-1, 0, -1):
		mpre[i-1]=max(mpre[i-1],mpre[i])
	ii = []
	for i in range(len(mrec)-1):
		if mrec[1:][i]!=mrec[0:-1][i]:
			ii.append(i+1)
	ap = 0
	for i in ii:
		ap = ap + np.sum((mrec[i]-mrec[i-1])*mpre[i])
# return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
	return [ap, mpre[0:len(mpre)-1], mrec[0:len(mpre)-1], ii]

