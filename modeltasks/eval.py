import numpy as np
import sys
from collections import Counter

class CFeval(object):
	"""docstring for CFeval"""
	def __init__(self, metrics, reshapeDims, classes):
		super(CFeval, self).__init__()
		self.metrics = metrics
		self.avgAcc  = []
		self.runThrough = False

	def evaluate(self, remoteOut, classValues):
		predictions = np.argmax(remoteOut, axis=1)
		self.avgAcc.append(np.sum(np.equal(predictions, classValues))/classValues.shape[0])

	def simRes(self):
		self.avgAcc = np.array(self.avgAcc)
		return np.mean(self.avgAcc)


class ODeval(object):
	"""docstring for ODeval"""
	def __init__(self, metrics, reshapeDims, classes):
		super(ODeval, self).__init__()
		self.metrics     = metrics
		self.iou         = metrics['map']['iou'] #iterate through for loop for multiple values
		self.reshapeDims = reshapeDims
		self.n_classes   = classes
		self.pred_format = {'class_id': 0, 'conf': 1, 'xmin': 2, 'ymin': 3, 'xmax': 4, 'ymax': 5}
		self.gt_format   = {'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}

		#pred format: class id, conf, xmin, ymin, xmax, ymax
		#ground truth: class id, xmin, ymin, xmax, ymax

		# The following lists all contain per-class data, i.e. all list have the length `n_classes + 1`,
		# where one element is for the background class, i.e. that element is just a dummy entry.
		self.prediction_results = [list() for _ in range(self.n_classes + 1)]
		self.num_gt_per_class = None
		# self.true_positives = None
		# self.false_positives = None
		# self.cumulative_true_positives = None
		# self.cumulative_false_positives = None
		# self.cumulative_precisions = None # "Cumulative" means that the i-th element in each list represents the precision for the first i highest condidence predictions for that class.
		# self.cumulative_recalls = None # "Cumulative" means that the i-th element in each list represents the recall for the first i highest condidence predictions for that class.
		# self.average_precisions = None
		# self.mean_average_precision = None
		# self.finalResults = dict(zip(self.iou, self.prediction_results))
		self.groundTruth = []
		self.imageId     = []
		self.runThrough = False

	def evaluate(self, remoteOut, labels):
		#ignore neutral boxes
		#num recall points=11
		#return precisions, recall, avg precisions
		groundTruth = labels[1]
		imageId     = labels[0]
		# print(imageId)
		if not self.runThrough:
			self.groundTruth+= list(groundTruth)
			[self.imageId.append(i) for i in imageId]
		# for i in self.iou:
		self.predictOnBatch( remoteOut, imageId, self.reshapeDims[0], self.reshapeDims[1])
		# print(len(self.prediction_results))

	def simRes(self):
		userRes = {}
		# print(len(self.groundTruth))
		# print(len(self.imageId))
		# num_gt_per_class = self.get_num_gt_per_class()
		# print(num_gt_per_class)
		# print(ions[7])
		# print(self.prediction_results[7])
		num_gt_per_class = 0
		for i in self.iou:
			userRes[i] = self.iterateOverIOU(self.prediction_results, i, num_gt_per_class, self.imageId)
		return np.array(list(userRes.items()))

	def iterateOverIOU(self, preds, iou, num_gt_per_class, imageId):
		return self.calcmAp(self.groundTruth, self.prediction_results, iou, imageId, self.n_classes)

	def predictOnBatch(self, remoteOut, imageId, imgH, imgW):
		class_id_pred = self.pred_format['class_id']
		conf_pred     = self.pred_format['conf']
		xmin_pred     = self.pred_format['xmin']
		ymin_pred     = self.pred_format['ymin']
		xmax_pred     = self.pred_format['xmax']
		ymax_pred     = self.pred_format['ymax']

		y_pred_filtered = []
		for i in range(len(remoteOut)):
			y_pred_filtered.append(remoteOut[i][remoteOut[i, :, 0] >=0.5])
		remoteOut = y_pred_filtered

		for k, batch_item in enumerate(remoteOut):
			image_id = imageId[k]

			for box in batch_item:
				class_id   = int(box[class_id_pred])
				confidence = box[conf_pred]
				xmin = round(box[xmin_pred], 1)
				ymin = round(box[ymin_pred], 1)
				xmax = round(box[xmax_pred], 1)
				ymax = round(box[ymax_pred], 1)
				prediction = (image_id, confidence, xmin, ymin, xmax, ymax)
				self.prediction_results[class_id].append(prediction)


	def calcmAp(self, labels, predictions, IOUThreshold, imageIds, n_classes):
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
				groundTruths.append(b)

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

