from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from tqdm import trange
import os
import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
from needle.data import *
from needle.init import *

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


# dice score
# def integral_dice(pred, gt, k):
#     '''
#     Dice coefficient for multiclass hard thresholded prediction consisting of 
#     integers instead of binary values. 
#     k = integer for class for which Dice is being calculated.
#     '''
#     return (torch.sum(pred[gt == k] == k)*2.0
#             / (torch.sum(pred[pred == k] == k)
#                + torch.sum(gt[gt == k] == k)).float())
	


def learn(model, loader, optimizer, process, device):
	""" main function for single epoch of train, val or test """
	dice_list = []
	running_loss = 0
	# num_batches = len(loader)
	num_batches = len(loader.dataset) // loader.batch_size
	with trange(num_batches, desc=process, ncols=100) as t:
		for batch_num, sample in enumerate(loader):
			print(f'batch_num: {batch_num}')
			img_batch, masks = sample
			# masks = masks[:, 0, :, :, 0].long()
			masks = np.array(masks)[:,:,:,0]
			# print(f'type(masks): {type(masks)}, masks.shape: {masks.shape}, masks.dtype:{masks.dtype}')
			image_batch, masks = ndl.Tensor(img_batch, device=device), ndl.Tensor(masks, device=device)
			# print(f'type(masks): {type(masks)}, masks.shape: {masks.shape}, masks.dtype:{masks.dtype}')
			# print(f'type(image_batch): {type(image_batch)}, image_batch.shape: {image_batch.shape}')
			# masks = masks[:, 0, :, :, 0].long()
			# print(type(masks), masks.shape, masks)
   
			# # one hot encoding labels
			# masks_oh = one_hot(masks, num_classes=2, device='cpu', dtype=masks.dtype)
   
			if process == 'train':
				model.train()
				optimizer.reset_grad()
	
				# preds = F.softmax(model(img_batch.cuda()), 1)
				# loss = F.binary_cross_entropy(preds, masks_oh.cuda())
	
				preds = model(image_batch)
				print(f'type(preds): {type(preds)}, preds.shape: {preds.shape}')
	# 			loss = nn.softmax(logits=preds, y=masks)
	# 			loss.backward()
	# 			optimizer.step()
	# 		else:
	# 			model.eval()
	# 			with torch.no_grad():
	# 				preds = F.softmax(model(img_batch.cuda()), 1)
	# 				loss = F.binary_cross_entropy(preds, masks_oh.cuda())
					
	# 		# hard_preds = torch.argmax(preds, 1)
	# 		# dice = integral_dice(hard_preds, masks, 1)
	# 		# dice_list.append(dice.item())
	# 		running_loss += loss
	# 		t.set_postfix(loss=running_loss.item()/(float(batch_num+1)*batch_size))
	# 		t.update()
	# mean_dice = np.mean(np.array(dice_list))
	# final_loss = running_loss.item()/(num_batches*batch_size)
	# return mean_dice, final_loss


def perform_learning(model, optimizer, path, all_names, batch_size,
					 splits, num_epochs, device):
	""" Wrapper function to run train, val, test loops """
	train_size, val_size, test_size = splits
	trn_names, val_names, tst_names = get_splits(all_names, train_size, val_size,
												test_size)
	train_loader, val_loader, test_loader = get_data_loaders(
		['train', 'val', 'test'],
		path, [trn_names, val_names, tst_names],
		augment=False,
		p=0.5,
		batch_size=batch_size
	)
	for epoch_num in range(num_epochs):
		# train set
		train_dice, train_loss = learn(model, train_loader, optimizer, 'train', device)
		print(f'Training Epoch {epoch_num} - Loss: {train_loss} ; Dice : {train_dice}')

	# 	# val set
	# 	val_dice, val_loss = learn(model, val_loader, optimizer, 'val')
	# 	print(f'Validation Epoch {epoch_num} - Loss: {val_loss} ; Dice : {val_dice}')
	
	# # test set
	# tst_dice, tst_loss = learn(model, test_loader, optimizer, 'test')
	# print(f'Test - Loss: {tst_loss} ; Dice : {tst_dice}')



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
	
	path = './data/us_dataset'
	all_names = os.listdir(os.path.join(path, 'all_images'))

	lr = 1e-4
	wt_dec = 1e-4
	num_epochs = 5
	batch_size = 2
	splits = [0.8, 0.1, 0.1]
	device = ndl.cuda()
	model = unet(n_classes=2, device=device, dtype="float32")
	# model = ResNet9(device=device)

	optimizer = ndl.optim.Adam(model.parameters(), lr=lr, weight_decay=wt_dec)

	perform_learning(model, optimizer, path, all_names, batch_size,
					splits, num_epochs, device)

	# qualitative_assessment()
