import os
import random
from PIL import Image
from torch.utils.data.dataset import Dataset


class EpisodeDataset(Dataset):
	def __init__(self, root, n_shots, n_ways, split='train', transform=None):
		self.__transform = transform
		self.__n_shots = n_shots
		self.__n_ways = n_ways
		self.__root = root

		self.__classes = self.__read_classes()
		self.__images = self.__read_images(split)

	def __read_classes(self):		
		with open(os.path.join(self.__root, 'classes.txt'), 'r+') as file:
			classes = [ class_id.strip() for class_id in file ]
		return classes

	def __read_images(self, split):
		split_path = os.path.join(self.__root, '{}.txt'.format(split))
		if not os.path.exists(split_path):
			raise ValueError('Split \'{}\' does not exists'.format(split))

		with open(split_path, 'r+') as file:
			files = [ os.path.join(self.__root, path.strip()) for path in file ]
		return files

	def __random_mapper(self, class_id):
		images = list(filter(lambda x: class_id in x, self.__images))
		return Image.open(random.choice(images))

	def __len__(self):
		return 1

	def __getitem__(self, x):
		random_classes = random.choices(self.__classes, k=self.__n_ways)

		x = list(map(self.__random_mapper, random_classes))
		y = random_classes.copy()

		if self.__transform is not None:
			x = list(map(self.__transform, x))

		return x, y

