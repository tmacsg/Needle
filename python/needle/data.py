import numpy as np
from PIL import Image
import cv2
from .autograd import Tensor
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd


class Transform:
	def __call__(self, x):
		raise NotImplementedError


class RandomFlipHorizontal(Transform):
	def __init__(self, p=0.5):
		self.p = p

	def __call__(self, img):
		"""
		Horizonally flip an image, specified as n H x W x C NDArray.
		Args:
			img: H x W x C NDArray of an image
		Returns:
			H x W x C ndarray corresponding to image flipped with probability self.p
		Note: use the provided code to provide randomness, for easier testing
		"""
		flip_img = np.random.rand() < self.p
		### BEGIN YOUR SOLUTION
		if flip_img:
			return np.flip(img, 1)
		else:
			return img
		### END YOUR SOLUTION


class RandomCrop(Transform):
	def __init__(self, padding=3):
		self.padding = padding

	def __call__(self, img):
		"""Zero pad and then randomly crop an image.
		Args:
			 img: H x W x C NDArray of an image
		Return
			H x W x C NAArray of cliped image
		Note: generate the image shifted by shift_x, shift_y specified below
		"""
		shift_x, shift_y = np.random.randint(
			low=-self.padding, high=self.padding + 1, size=2
		)
		### BEGIN YOUR SOLUTION
		img_padded = np.pad(img, ((self.padding,self.padding), (self.padding,self.padding), (0,0)))
		start_x = self.padding + shift_x
		start_y = self.padding + shift_y
		return img_padded[start_x:start_x+img.shape[0], start_y:start_y+img.shape[1], :]
		### END YOUR SOLUTION


class Dataset:
	r"""An abstract class representing a `Dataset`.

	All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
	data sample for a given key. Subclasses must also overwrite
	:meth:`__len__`, which is expected to return the size of the dataset.
	"""

	def __init__(self, transforms: Optional[List] = None):
		self.transforms = transforms

	def __getitem__(self, index) -> object:
		raise NotImplementedError

	def __len__(self) -> int:
		raise NotImplementedError

	def apply_transforms(self, x):
		if self.transforms is not None:
			# apply the transforms
			for tform in self.transforms:
				x = tform(x)
		return x


class DataLoader:
	r"""
	Data loader. Combines a dataset and a sampler, and provides an iterable over
	the given dataset.
	Args:
		dataset (Dataset): dataset from which to load the data.
		batch_size (int, optional): how many samples per batch to load
			(default: ``1``).
		shuffle (bool, optional): set to ``True`` to have the data reshuffled
			at every epoch (default: ``False``).
	"""
	dataset: Dataset
	batch_size: Optional[int]

	def __init__(
		self,
		dataset: Dataset,
		batch_size: Optional[int] = 1,
		shuffle: bool = False,
	):

		self.dataset = dataset
		self.shuffle = shuffle
		self.batch_size = batch_size
		if not self.shuffle:
			self.ordering = np.array_split(
				np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
			)

	def __iter__(self):
		### BEGIN YOUR SOLUTION
		if self.shuffle:
			self.ordering = np.array_split(np.random.permutation(len(self.dataset)), 
				range(self.batch_size, len(self.dataset), self.batch_size))
		self.index = 0
		### END YOUR SOLUTION
		return self

	def __next__(self):
		### BEGIN YOUR SOLUTION
		if self.index == len(self.ordering):
			raise StopIteration

		indexes = self.ordering[self.index]
		self.index += 1
	   
		# self.dataset[0] is tuple (X, )

		if len(self.dataset[0]) == 1:
			if self.batch_size == 1:
				# batch_data = Tensor(self.dataset[indexes[0]][0])
				batch_data = self.dataset[indexes[0]][0]
				batch_data = batch_data.reshape((1, *batch_data.shape))
				return (batch_data, )
			else:
				batch_x = []
				for i in range(len(indexes)):
					batch_x.append(self.dataset[indexes[i]][0])
				return (batch_x, )
				# return (Tensor(batch_x), )
		# self.dataset[0] is tuple (X, Y)
		else:
			if self.batch_size == 1:
				# batch_x = Tensor(self.dataset[indexes[0]][0])
				batch_x = self.dataset[indexes[0]][0]
				batch_x = batch_x.reshape((1, *batch_x.shape))
				# print(batch_x.shape)
				return batch_x, self.dataset[indexes[0]][1]
				# return batch_x, Tensor(self.dataset[indexes[0]][1])
			else:
				batch_x = []
				batch_y = []
				for i in range(len(indexes)):
					batch_x.append(self.dataset[indexes[i]][0])
					batch_y.append(self.dataset[indexes[i]][1])            
				return batch_x, batch_y
				# return Tensor(batch_x), Tensor(batch_y)
		### END YOUR SOLUTION


class MNISTDataset(Dataset):
	def __init__(
		self,
		image_filename: str,
		label_filename: str,
		transforms: Optional[List] = None,
	):
		### BEGIN YOUR SOLUTION
		super().__init__(transforms)
		imageSet, labelSet = parse_mnist(image_filename, label_filename)
		self.imageSet = np.reshape(imageSet, (imageSet.shape[0], 28, 28, 1))
		self.labelSet = labelSet
		### END YOUR SOLUTION

	def __getitem__(self, index) -> object:
		### BEGIN YOUR SOLUTION
		imageData = self.imageSet[index]       
		imageData = self.apply_transforms(imageData)
		return imageData, self.labelSet[index]
		### END YOUR SOLUTION

	def __len__(self) -> int:
		### BEGIN YOUR SOLUTION
		return self.imageSet.shape[0]
		### END YOUR SOLUTION



class CIFAR10Dataset(Dataset):
	def __init__(
		self,
		base_folder: str,
		train: bool,
		p: Optional[int] = 0.5,
		transforms: Optional[List] = None
	):
		"""
		Parameters:
		base_folder - cifar-10-batches-py folder filepath
		train - bool, if True load training dataset, else load test dataset
		Divide pixel values by 255. so that images are in 0-1 range.
		Attributes:
		X - numpy array of images
		y - numpy array of labels
		"""
		### BEGIN YOUR SOLUTION
		super().__init__(transforms)
		self.train = train
		self.p = p

		X = []
		y = []
		if train:
			for i in range(1,6):
				dic = self.unpickle(os.path.join(base_folder, f'data_batch_{i}'))
				X.append(np.vstack(dic['data'.encode()]))
				y.extend(dic['labels'.encode()])
		else:
			dic = self.unpickle(os.path.join(base_folder, f'test_batch')) 
			X = dic['data'.encode()]
			y = dic['labels'.encode()]

		X = np.array(X) / 255.0
		X = X.reshape((-1,3,32,32))
		self.X = X
		self.y = np.array(y)

		### END YOUR SOLUTION

	def __getitem__(self, index) -> object:
		"""
		Returns the image, label at given index
		Image should be of shape (3, 32, 32)
		"""
		### BEGIN YOUR SOLUTION
		image = self.X[index]
		label = self.y[index]
		image = self.apply_transforms(image)
		return image, label
		### END YOUR SOLUTION

	def __len__(self) -> int:
		"""
		Returns the total number of examples in the dataset
		"""
		### BEGIN YOUR SOLUTION
		return len(self.X)
		### END YOUR SOLUTION

	@staticmethod
	def unpickle(file):
		import pickle
		with open(file, 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
		return dict


class FetalHeadDataset(Dataset):
	""" Data loader class """
	def __init__(
     self, 
     path: str, 
     file_list: List[str], 
     transforms: Optional[List] = None,
     p: Optional[int]=None
     ):
		"""
		Args:
			path (str): path where images stored
			file_list (List[str]): list of images in current split
			transforms (List[str]): list of transforms
			p (float): Probability of applying random aug (if transforms != None)
		"""
		self.path = path
		self.file_list = file_list
		self.transforms = transforms
		self.p = p

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, idx):
		""" Preprocess and return a single sample & label """
		img_name = os.path.join(self.path, 'all_images', self.file_list[idx])
		mask_fname = self.file_list[idx].split('.')[0] + '_mask.png'
		mask_name = os.path.join(self.path, 'all_masks', mask_fname)
		img = Image.open(img_name)
		mask = Image.open(mask_name)

		# Resize to dimensions supported by Vanilla UNet
		img = img.resize((572, 572), Image.LANCZOS)
		mask = mask.resize((388, 388), Image.NEAREST)

		img = np.array(img)
		mask = np.array(mask)
		mask[mask == 255] = 1

		# mask = Tensor([mask])

		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		# img = Tensor(img)
		img = img.transpose(2, 0, 1)

		# select and apply random augmentation (if passed)
		if self.transforms:
			do_aug = np.random.choice([True, False], 1, p=[self.p,
															1-self.p])
			if do_aug:
				aug_name = np.random.choice(self.transforms, 1)
				img = aug_name[0](img)
		# img = (img - torch.mean(img)) / torch.std(img)
		img = (img - np.mean(img)) / np.std(img)
		return img, list(mask)
	


class NDArrayDataset(Dataset):
	def __init__(self, *arrays):
		self.arrays = arrays

	def __len__(self) -> int:
		return self.arrays[0].shape[0]

	def __getitem__(self, i) -> object:
		return tuple([a[i] for a in self.arrays])






class Dictionary(object):
	"""
	Creates a dictionary from a list of words, mapping each word to a
	unique integer.
	Attributes:
	word2idx: dictionary mapping from a word to its unique ID
	idx2word: list of words in the dictionary, in the order they were added
		to the dictionary (i.e. each word only appears once in this list)
	"""
	def __init__(self):
		self.word2idx = {}
		self.idx2word = []

	def add_word(self, word):
		"""
		Input: word of type str
		If the word is not in the dictionary, adds the word to the dictionary
		and appends to the list of words.
		Returns the word's unique ID.
		"""
		### BEGIN YOUR SOLUTION
		raise NotImplementedError()
		### END YOUR SOLUTION

	def __len__(self):
		"""
		Returns the number of unique words in the dictionary.
		"""
		### BEGIN YOUR SOLUTION
		raise NotImplementedError()
		### END YOUR SOLUTION



class Corpus(object):
	"""
	Creates corpus from train, and test txt files.
	"""
	def __init__(self, base_dir, max_lines=None):
		self.dictionary = Dictionary()
		self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
		self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

	def tokenize(self, path, max_lines=None):
		"""
		Input:
		path - path to text file
		max_lines - maximum number of lines to read in
		Tokenizes a text file, first adding each word in the file to the dictionary,
		and then tokenizing the text file to a list of IDs. When adding words to the
		dictionary (and tokenizing the file content) '<eos>' should be appended to
		the end of each line in order to properly account for the end of the sentence.
		Output:
		ids: List of ids
		"""
		### BEGIN YOUR SOLUTION
		raise NotImplementedError()
		### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
	"""
	Starting from sequential data, batchify arranges the dataset into columns.
	For instance, with the alphabet as the sequence and batch size 4, we'd get
	┌ a g m s ┐
	│ b h n t │
	│ c i o u │
	│ d j p v │
	│ e k q w │
	└ f l r x ┘.
	These columns are treated as independent by the model, which means that the
	dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
	batch processing.
	If the data cannot be evenly divided by the batch size, trim off the remainder.
	Returns the data as a numpy array of shape (nbatch, batch_size).
	"""
	### BEGIN YOUR SOLUTION
	raise NotImplementedError()
	### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
	"""
	get_batch subdivides the source data into chunks of length bptt.
	If source is equal to the example output of the batchify function, with
	a bptt-limit of 2, we'd get the following two Variables for i = 0:
	┌ a g m s ┐ ┌ b h n t ┐
	└ b h n t ┘ └ c i o u ┘
	Note that despite the name of the function, the subdivison of data is not
	done along the batch dimension (i.e. dimension 1), since that was handled
	by the batchify function. The chunks are along dimension 0, corresponding
	to the seq_len dimension in the LSTM or RNN.
	Inputs:
	batches - numpy array returned from batchify function
	i - index
	bptt - Sequence length
	Returns:
	data - Tensor of shape (bptt, bs) with cached data as NDArray
	target - Tensor of shape (bptt*bs,) with cached data as NDArray
	"""
	### BEGIN YOUR SOLUTION
	raise NotImplementedError()
	### END YOUR SOLUTION


def parse_mnist(image_filesname, label_filename):
	import gzip
	with gzip.open(label_filename, 'rb') as trainLbpath:
		trainLabel = np.frombuffer(trainLbpath.read(), dtype=np.uint8, offset=8)
	with gzip.open(image_filesname, 'rb') as trainSetpath:
		trainSet = np.frombuffer(trainSetpath.read(), dtype=np.uint8,offset=16).reshape(len(trainLabel), 784)   
	trainSet = trainSet.astype(np.float32) / 255.0
	return trainSet, trainLabel