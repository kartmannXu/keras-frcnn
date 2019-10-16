import cv2
import numpy as np
import copy
import imgaug as ia
from imgaug import augmenters as iaa


def _augment(img_data, config, augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)

	img = cv2.imread(img_data_aug['filepath'])

	if augment:
		rows, cols = img.shape[:2]

		if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 1)
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2

		if config.use_vertical_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 0)
			for bbox in img_data_aug['bboxes']:
				y1 = bbox['y1']
				y2 = bbox['y2']
				bbox['y2'] = rows - y1
				bbox['y1'] = rows - y2

		if config.rot_90:
			angle = np.random.choice([0,90,180,270],1)[0]
			if angle == 270:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 0)
			elif angle == 180:
				img = cv2.flip(img, -1)
			elif angle == 90:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 1)
			elif angle == 0:
				pass

			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				y1 = bbox['y1']
				y2 = bbox['y2']
				if angle == 270:
					bbox['x1'] = y1
					bbox['x2'] = y2
					bbox['y1'] = cols - x2
					bbox['y2'] = cols - x1
				elif angle == 180:
					bbox['x2'] = cols - x1
					bbox['x1'] = cols - x2
					bbox['y2'] = rows - y1
					bbox['y1'] = rows - y2
				elif angle == 90:
					bbox['x1'] = rows - y2
					bbox['x2'] = rows - y1
					bbox['y1'] = x1
					bbox['y2'] = x2        
				elif angle == 0:
					pass

	img_data_aug['width'] = img.shape[1]
	img_data_aug['height'] = img.shape[0]
	return img_data_aug, img


def augment(img_data, config, augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)
	image = cv2.imread(img_data_aug['filepath'])
	if augment:
		seqs = [
			iaa.Multiply((0.8, 1.5)),
			iaa.GaussianBlur(sigma=(0.0, 3.0))
		]
		if config.use_horizontal_flips:
			seqs += [iaa.Fliplr(0.5)]
		if config.use_vertical_flips:
			seqs += [iaa.Flipud(0.5)]
		if config.rot_90:
			seqs += [iaa.OneOf([iaa.Affine(rotate=90),
					 iaa.Affine(rotate=90),
					 iaa.Affine(rotate=270),
					 iaa.Affine(rotate=180),
					 iaa.Affine(rotate=180),
					 iaa.Affine(rotate=270)])]

		augmentation = iaa.SomeOf((0, 3), seqs)

		temp_aug_bbox = []
		for bbox in img_data_aug["bboxes"]:
			temp_aug_bbox.append(ia.BoundingBox(x1=bbox["x1"],
												x2=bbox["x2"],
												y1=bbox["y1"],
												y2=bbox["y2"],
												label=bbox["class"]))
		bbs = ia.BoundingBoxesOnImage(temp_aug_bbox, shape=image.shape)


		seq_det = augmentation.to_deterministic()

		image = seq_det.augment_image(image)
		bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

		bboxes_resize1 = []
		for one in bbs_aug.bounding_boxes:
			bboxes_resize1.append({
				"class": one.label,
				"x1": int(one.x1),
				"x2": int(one.x2),
				"y1": int(one.y1),
				"y2": int(one.y2),
				"difficult": False
			})
		img_data_aug["bboxes"] = bboxes_resize1

	return img_data_aug, image