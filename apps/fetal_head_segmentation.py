from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
from needle.data import *

def get_data_loaders(categories, path, file_lists,
					 augment, p, batch_size):
	"""
	Wrapper function to return dataloader(s)
	Args:
	categories (List[str]): names of processes for which dataloader needed
	path (str): path where images stored
	file_lists (List[List[str]]): list of file lists
	augment (boolean): whether to apply augmentation
	p (float): Probability of applying random aug
	batch_size (int): batch size
	Returns:
	torch.utils.data.DataLoader object
	"""
	loaders = []
	for i, category in enumerate(categories):
		if category == 'train' and augment:
			transforms = [
				RandomFlipHorizontal(p=1)
			]
		else:
			transforms = None
		loader = DataLoader(
			FetalHeadDataset(path, file_lists[i], transforms, p),
			batch_size,
			shuffle=False
		)
		loaders.append(loader)
	return loaders



def get_splits(all_names, train_size, val_size, test_size):
	split1_size = (val_size+test_size)
	split2_size = test_size / (val_size+test_size)
	trn_names, valtst_names = train_test_split(
		all_names, test_size=split1_size, random_state=0)
	val_names, tst_names = train_test_split(
		valtst_names, test_size=split2_size, random_state=0)
	return trn_names, val_names, tst_names 



def qualitative_assessment():
	path = '../data/us_dataset'
	all_names = os.listdir(os.path.join(path, 'all_images'))

	splits = [0.8, 0.1, 0.1]
	train_size, val_size, test_size = splits
	trn_names, val_names, tst_names = get_splits(all_names, train_size, val_size,
												test_size)

	_, _, test_loader = get_data_loaders(
		['train', 'val', 'test'],
		path, [trn_names, val_names, tst_names],
		augment=False,
		p=0.5,
		batch_size=1
	)

	cnt = 1

	for batch_num, sample in enumerate(test_loader):
		X, label = sample

		# preds = F.softmax(model(X.cuda()), 1)
		# hard_preds = torch.argmax(preds, 1)
		# mask = hard_preds[0].detach().cpu().numpy()

		# print(type(X), X, label)

		# img = X[0].permute(1, 2, 0).numpy()
		img = X[0].transpose(1, 2, 0)
		img = img.astype('float')
		img = cv2.resize(img, (388, 388))
		img = (img - np.max(img)) / (np.max(img) - np.min(img))

		plt.figure()
		plt.subplot(1, 2, 1)
		plt.imshow((img*255).astype('uint8'), cmap='gray')
		plt.axis('off')

		plt.subplot(1, 2, 2)
		plt.imshow((img*255).astype('uint8'), cmap='gray')
		# plt.imshow(mask, cmap='jet', alpha=0.3)

		#   plt.title('Overlayed Attributions')
		plt.axis('off')
		#   plt.colorbar()
		plt.show()
		cnt += 1
		if cnt >= 5:
			break

if __name__ == "__main__":
	qualitative_assessment()
	
