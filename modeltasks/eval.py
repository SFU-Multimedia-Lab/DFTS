import numpy as np
from .utils import iou
from .tempeval import calcmAp

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
		return calcmAp(self.groundTruth, self.prediction_results, iou, imageId, self.n_classes)

		# true_positives, false_positives, cumulative_true_positives, cumulative_false_positives = self.match_predictions(preds, matching_iou_threshold=iou)
		# cumulative_precisions, cumulative_recalls = self.compute_precision_recall(num_gt_per_class, cumulative_true_positives, cumulative_false_positives)
		# average_precisions = self.compute_average_precions(cumulative_precisions, cumulative_recalls)
		# mean_average_precision = self.compute_mean_average_precision(average_precisions)
		# print(average_precisions)
		# return mean_average_precision



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


	# def get_num_gt_per_class(self):
	# 	#mkae it run once over all epochs
	# 	num_gt_per_class = np.zeros(shape=(self.n_classes+1), dtype=np.int)
	# 	class_id_index = self.gt_format['class_id']
	# 	ground_truth = self.groundTruth

	# 	for i in range(len(ground_truth)):
	# 		boxes = np.asarray(ground_truth[i])

	# 		for j in range(boxes.shape[0]):
	# 			class_id = int(boxes[j, class_id_index])
	# 			# print(boxes)
	# 			# print(class_id)
	# 			# print(type(class_id))
	# 			num_gt_per_class[class_id] += 1
	# 	return num_gt_per_class

	# def match_predictions(self, preds, matching_iou_threshold=0.5):
	# 	class_id_gt = self.gt_format['class_id']
	# 	xmin_gt = self.gt_format['xmin']
	# 	ymin_gt = self.gt_format['ymin']
	# 	xmax_gt = self.gt_format['xmax']
	# 	ymax_gt = self.gt_format['ymax']

	# 	ground_truth = {}

	# 	for i in range(len(self.imageId)):
	# 		image_id = str(self.imageId[i])
	# 		labels   = self.groundTruth[i]
	# 		ground_truth[image_id] = np.asarray(labels)

	# 	true_positives = [[]] # The false positives for each class, sorted by descending confidence.
	# 	false_positives = [[]] # The true positives for each class, sorted by descending confidence.
	# 	cumulative_true_positives = [[]]
	# 	cumulative_false_positives = [[]]

	# 	for class_id in range(1, self.n_classes+1):
	# 		predictions = preds[class_id]

	# 		# Store the matching results in these lists:
	# 		true_pos = np.zeros(len(predictions), dtype=np.int) # 1 for every prediction that is a true positive, 0 otherwise
	# 		false_pos = np.zeros(len(predictions), dtype=np.int) # 1 for every prediction that is a false positive, 0 otherwise

	# 		# In case there are no predictions at all for this class, we're done here.
	# 		if len(predictions) == 0:
	# 			print("No predictions for class {}/{}".format(class_id, self.n_classes))
	# 			true_positives.append(true_pos)
	# 			false_positives.append(false_pos)
	# 			continue

	# 		# Convert the predictions list for this class into a structured array so that we can sort it by confidence.

	# 		# Get the number of characters needed to store the image ID strings in the structured array.
	# 		num_chars_per_image_id = len(str(predictions[0][0])) + 6 # Keep a few characters buffer in case some image IDs are longer than others.
	# 		# Create the data type for the structured array.
	# 		preds_data_type = np.dtype([('image_id', 'U{}'.format(num_chars_per_image_id)),
	# 									('confidence', 'f4'),
	# 									('xmin', 'f4'),
	# 									('ymin', 'f4'),
	# 									('xmax', 'f4'),
	# 									('ymax', 'f4')])
	# 		# Create the structured array
	# 		predictions = np.array(predictions, dtype=preds_data_type)

	# 		# Sort the detections by decreasing confidence.
	# 		descending_indices = np.argsort(-predictions['confidence'], kind='quicksort')
	# 		predictions_sorted = predictions[descending_indices]

	# 		# Keep track of which ground truth boxes were already matched to a detection.
	# 		gt_matched = {}

	# 		# print(len(predictions.shape))
	# 		# print(len(predictions))
	# 		# print(predictions[:]['image_id'])

	# 		for i in range(len(predictions)):
	# 			prediction = predictions_sorted[i]
	# 			image_id = prediction['image_id']
	# 			pred_box = np.asarray(list(prediction[['xmin', 'ymin', 'xmax', 'ymax']])) # Convert the structured array element to a regular array.

	# 			# Get the relevant ground truth boxes for this prediction,
	# 			# i.e. all ground truth boxes that match the prediction's
	# 			# image ID and class ID.

	# 			# The ground truth could either be a tuple with `(ground_truth_boxes, eval_neutral_boxes)`
	# 			# or only `ground_truth_boxes`.

	# 			gt = ground_truth[image_id]
	# 			gt = np.asarray(gt)
	# 			class_mask = gt[:,class_id_gt] == class_id
	# 			gt = gt[class_mask]

	# 			if gt.size == 0:
	# 				# If the image doesn't contain any objects of this class,
	# 				# the prediction becomes a false positive.
	# 				false_pos[i] = 1
	# 				continue

	# 			# Compute the IoU of this prediction with all ground truth boxes of the same class.
	# 			overlaps = iou(boxes1=gt[:,[xmin_gt, ymin_gt, xmax_gt, ymax_gt]],
	# 						   boxes2=pred_box,
	# 						   coords='corners',
	# 						   mode='element-wise')

	# 			# For each detection, match the ground truth box with the highest overlap.
	# 			# It's possible that the same ground truth box will be matched to multiple
	# 			# detections.
	# 			gt_match_index = np.argmax(overlaps)
	# 			gt_match_overlap = overlaps[gt_match_index]

	# 			if gt_match_overlap < matching_iou_threshold:
	# 				# False positive, IoU threshold violated:
	# 				# Those predictions whose matched overlap is below the threshold become
	# 				# false positives.
	# 				false_pos[i] = 1
	# 			else:
	# 				if not (image_id in gt_matched):
	# 						# True positive:
	# 						# If the matched ground truth box for this prediction hasn't been matched to a
	# 						# different prediction already, we have a true positive.
	# 						true_pos[i] = 1
	# 						gt_matched[image_id] = np.zeros(shape=(gt.shape[0]), dtype=np.bool)
	# 						gt_matched[image_id][gt_match_index] = True
	# 				elif not gt_matched[image_id][gt_match_index]:
	# 						# True positive:
	# 						# If the matched ground truth box for this prediction hasn't been matched to a
	# 						# different prediction already, we have a true positive.
	# 						true_pos[i] = 1
	# 						gt_matched[image_id][gt_match_index] = True
	# 				else:
	# 						# False positive, duplicate detection:
	# 						# If the matched ground truth box for this prediction has already been matched
	# 						# to a different prediction previously, it is a duplicate detection for an
	# 						# already detected object, which counts as a false positive.
	# 						false_pos[i] = 1

	# 		true_positives.append(true_pos)
	# 		false_positives.append(false_pos)

	# 		cumulative_true_pos = np.cumsum(true_pos) # Cumulative sums of the true positives
	# 		cumulative_false_pos = np.cumsum(false_pos) # Cumulative sums of the false positives

	# 		cumulative_true_positives.append(cumulative_true_pos)
	# 		cumulative_false_positives.append(cumulative_false_pos)

	# 	return true_positives, false_positives, cumulative_true_positives, cumulative_false_positives

	# def compute_precision_recall(self, num_gt_per_class, cumulative_true_positives, cumulative_false_positives):
	# 	cumulative_precisions = [[]]
	# 	cumulative_recalls = [[]]

	# 	for class_id in range(1, self.n_classes + 1):
	# 		tp = cumulative_true_positives[class_id]
	# 		fp = cumulative_false_positives[class_id]

	# 		print(type(tp))
	# 		print(type(fp))

	# 		cumulative_precision = np.where(tp + fp > 0, tp / (tp + fp), 0) # 1D array with shape `(num_predictions,)`
	# 		cumulative_recall = tp / num_gt_per_class[class_id] # 1D array with shape `(num_predictions,)`
	# 		cumulative_precisions.append(cumulative_precision)
	# 		cumulative_recalls.append(cumulative_recall)

	# 	return cumulative_precisions, cumulative_recalls

	# def compute_average_precions(self, cumulative_precisions, cumulative_recalls):
	# 	average_precisions = [0.0]
	# 	num_recall_points  = 11

	# 	for class_id in range(1, self.n_classes + 1):
	# 		cumulative_precision = cumulative_precisions[class_id]
	# 		cumulative_recall = cumulative_recalls[class_id]
	# 		average_precision = 0.0

	# 		for t in np.linspace(start=0, stop=1, num=11, endpoint=True):
	# 			cum_prec_recall_greater_t = cumulative_precision[cumulative_recall >= t]
	# 			if cum_prec_recall_greater_t.size == 0:
	# 				precision = 0.0
	# 			else:
	# 				precision = np.amax(cum_prec_recall_greater_t)
	# 			average_precision += precision
	# 		average_precision /= num_recall_points
	# 		average_precisions.append(average_precision)

	# 	return average_precisions

	# def compute_mean_average_precision(self, average_precisions):
	# 	mean_average_precision = np.mean(average_precisions[1:])
	# 	return mean_average_precision

