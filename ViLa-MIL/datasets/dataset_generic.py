from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)
	print()

class Generic_WSI_Classification_Dataset(Dataset):
	def __init__(self,
				 csv_path = 'dataset_csv/ccrcc_clean.csv',
				 mode = 'clam',
				 shuffle = False,
				 seed = 7,
				 print_info = True,
				 label_dict = {},
				 filter_dict = {},
				 ignore=[],
				 patient_strat=False,
				 label_col = None,
				 patient_voting = 'max',
				 ):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
		"""
		self.label_dict = label_dict
		self.num_classes = len(set(self.label_dict.values()))
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir_s = None
		self.data_dir_l = None
		if not label_col:
			label_col = 'label'
		self.label_col = label_col

		slide_data = pd.read_csv(csv_path)
		slide_data = self.filter_df(slide_data, filter_dict)
		slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)

		self.slide_data = slide_data

		self.patient_data_prep(patient_voting)
		self.mode = mode
		self.cls_ids_prep()

		if print_info:
			self.summarize()

	def cls_ids_prep(self):
		self.patient_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id'])) 
		patient_labels = []

		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			label = self.slide_data['label'][locations].values
			if patient_voting == 'max':
				label = label.max() 
			elif patient_voting == 'maj':
				label = stats.mode(label)[0]
			else:
				raise NotImplementedError
			patient_labels.append(label)

		self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

	@staticmethod
	def df_prep(data, label_dict, ignore, label_col):
		if label_col != 'label':
			data['label'] = data[label_col].copy()

		mask = data['label'].isin(ignore)
		data = data[~mask]
		data.reset_index(drop=True, inplace=True)
		for i in data.index:
			key = data.loc[i, 'label']
			data.at[i, 'label'] = label_dict[key]

		return data

	def filter_df(self, df, filter_dict={}):
		if len(filter_dict) > 0:
			filter_mask = np.full(len(df), True, bool)
			for key, val in filter_dict.items():
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
		return df

	def __len__(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)

	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))
		print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
		for i in range(self.num_classes):
			print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
			print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

	def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
		settings = {
			'n_splits' : k,
			'val_num' : val_num,
			'test_num': test_num,
			'label_frac': label_frac,
			'seed': self.seed,
			'custom_test_ids': custom_test_ids
		}

		if self.patient_strat:
			settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
		else:
			settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

		self.split_gen = generate_split(**settings)

	def set_splits(self,start_from=None):
		if start_from:
			ids = nth(self.split_gen, start_from)

		else:
			ids = next(self.split_gen)

		if self.patient_strat:
			slide_ids = [[] for i in range(len(ids))]

			for split in range(len(ids)):
				for idx in ids[split]:
					case_id = self.patient_data['case_id'][idx]
					slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
					slide_ids[split].extend(slide_indices)

			self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

		else:
			self.train_ids, self.val_ids, self.test_ids = ids

	def get_split_from_df(self, all_splits, split_key='train'):
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, mode=self.mode, num_classes=self.num_classes)
		else:
			split = None

		return split

	def get_merged_split_from_df(self, all_splits, split_keys=['train']):
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(merged_split)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, mode=self.mode, num_classes=self.num_classes)
		else:
			split = None

		return split

	def return_splits(self, from_id=True, csv_path=None):

		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split(train_data, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, mode=self.mode,  num_classes=self.num_classes)

			else:
				train_split = None

			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, mode=self.mode,  num_classes=self.num_classes)

			else:
				val_split = None

			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(test_data, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, num_classes=self.num_classes)

			else:
				test_split = None


		else:
			assert csv_path
			all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  
			train_split = self.get_split_from_df(all_splits, 'train')  
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test') 

		return train_split, val_split, test_split

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids):
		return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):

		if return_descriptor:
			index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
			columns = ['train', 'val', 'test']
			df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							  columns= columns)

		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		labels = self.getlabel(self.train_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'train'] = counts[u]

		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		labels = self.getlabel(self.val_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'val'] = counts[u]

		count = len(self.test_ids)
		print('\nnumber of test samples: {}'.format(count))
		labels = self.getlabel(self.test_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'test'] = counts[u]

		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		# assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

		if return_descriptor:
			return df

	def save_split(self, filename):
		train_split = self.get_list(self.train_ids)
		val_split = self.get_list(self.val_ids)
		test_split = self.get_list(self.test_ids)
		df_tr = pd.DataFrame({'train': train_split})
		df_v = pd.DataFrame({'val': val_split})
		df_t = pd.DataFrame({'test': test_split})
		df = pd.concat([df_tr, df_v, df_t], axis=1)
		df.to_csv(filename, index = False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
				 data_dir_s,
				 data_dir_l,
				 mode,
				 **kwargs):

		super(Generic_MIL_Dataset, self).__init__(**kwargs)
		self.data_dir_s = data_dir_s
		self.data_dir_l = data_dir_l
		self.mode = mode
		self.use_h5 = False
		self._h5_index_s = None
		self._h5_index_l = None
		# If using uploaded features, the directory may only contain a subset of slides.
		# Filter out rows that don't have both 20x/10x features to avoid crashing mid-epoch.
		try:
			if (
				self.mode == 'transformer'
				and isinstance(self.data_dir_s, str)
				and isinstance(self.data_dir_l, str)
				and os.path.isdir(self.data_dir_s)
				and os.path.isdir(self.data_dir_l)
			):
				self._h5_index_s = self._build_h5_index(self.data_dir_s, expect_10x=False)
				self._h5_index_l = self._build_h5_index(self.data_dir_l, expect_10x=True)
				orig_len = len(self.slide_data)
				keep = []
				for i in range(len(self.slide_data)):
					sid = self.slide_data['slide_id'][i]
					key = self._normalize_slide_id(sid)
					if key in self._h5_index_s and key in self._h5_index_l:
						keep.append(i)
				if len(keep) < orig_len:
					self.slide_data = self.slide_data.loc[keep].reset_index(drop=True)
					# parent __init__ already built these based on original slide_data; rebuild.
					self.patient_data_prep('max')
					self.cls_ids_prep()
					print(f"[Generic_MIL_Dataset] Filtered slide_data by available .h5: {len(keep)}/{orig_len}")
		except Exception:
			# Fall back to original behavior if anything unexpected happens.
			pass

	def _normalize_slide_id(self, slide_id: str) -> str:
		"""
		Normalize slide ids and uploaded feature filenames to a comparable key.
		Examples:
		- TCGA-xx.svs            -> TCGA-xx
		- uuid__TCGA-xx.h5       -> TCGA-xx
		- TCGA-xx_10x.h5         -> TCGA-xx
		- TCGA-xx.svs.h5         -> TCGA-xx
		"""
		s = str(slide_id)
		if "__" in s:
			s = s.split("__", 1)[1]
		if s.endswith(".h5"):
			s = s[:-3]
		if s.endswith("_10x"):
			s = s[:-4]
		if s.endswith(".svs"):
			s = s[:-4]
		return s

	def _build_h5_index(self, data_dir: str, expect_10x: bool) -> dict[str, str]:
		index: dict[str, str] = {}
		try:
			for fn in os.listdir(data_dir):
				if not fn.endswith(".h5"):
					continue
				key = self._normalize_slide_id(fn)
				# for 10x we may store as *_10x.h5; still normalize to base
				index[key] = os.path.join(data_dir, fn)
		except Exception:
			pass
		return index

	def _resolve_h5_path(self, data_dir: str, slide_id: str, scale: str) -> str:
		"""
		Resolve a slide_id to a real .h5 path in data_dir.
		Tries exact '<slide_id>.h5' first (original behavior), then falls back to
		normalized matching against uploaded filenames (uuid__prefix, .svs, _10x).
		"""
		# 1) original expected path
		candidate = os.path.join(data_dir, "{}.h5".format(slide_id))
		if os.path.isfile(candidate):
			return candidate

		# 2) common fallback: drop '.svs' from slide_id
		if str(slide_id).endswith(".svs"):
			candidate2 = os.path.join(data_dir, "{}.h5".format(str(slide_id)[:-4]))
			if os.path.isfile(candidate2):
				return candidate2

		# 3) indexed lookup
		key = self._normalize_slide_id(slide_id)
		if scale == "s":
			if self._h5_index_s is None:
				self._h5_index_s = self._build_h5_index(data_dir, expect_10x=False)
			if key in self._h5_index_s:
				return self._h5_index_s[key]
		else:
			if self._h5_index_l is None:
				self._h5_index_l = self._build_h5_index(data_dir, expect_10x=True)
			if key in self._h5_index_l:
				return self._h5_index_l[key]

		# 4) as a last resort, try matching base without any suffix
		raise FileNotFoundError(candidate)

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'][idx]
		label = self.slide_data['label'][idx]
		if type(self.data_dir_s) == dict and type(self.data_dir_l) == dict:
			source = self.slide_data['source'][idx]
			data_dir_s = self.data_dir_s['source']
			data_dir_l = self.data_dir_l['source']
		else:
			data_dir_s = self.data_dir_s
			data_dir_l = self.data_dir_l

		if not self.use_h5:
			if self.data_dir_s and self.data_dir_l:
				if(self.mode == 'transformer'):
					path_s = self._resolve_h5_path(data_dir_s, slide_id, scale="s")
					path_l = self._resolve_h5_path(data_dir_l, slide_id, scale="l")
					scale_s = h5py.File(path_s, 'r')
					scale_l = h5py.File(path_l, 'r')
					# h5 通常为 float64，这里统一转 float32，避免与模型权重 dtype 不匹配
					features_s = torch.from_numpy(np.array(scale_s['features'])).float()
					coords_s = torch.from_numpy(np.array(scale_s['coords'])).float()
					features_l = torch.from_numpy(np.array(scale_l['features'])).float()
					coords_l = torch.from_numpy(np.array(scale_l['coords'])).float()
					return features_s, coords_s, features_l, coords_l, label

			else:
				return slide_id, label

		else:
			full_path = os.path.join(data_dir_s, 'h5_files', '{}.h5'.format(slide_id))
			with h5py.File(full_path,'r') as hdf5_file:
				features = hdf5_file['features'][:]
				coords = hdf5_file['coords'][:]

			features = torch.from_numpy(features)
			return features, label, coords


class Generic_Split(Generic_MIL_Dataset):
	def __init__(self, slide_data, data_dir_s=None, data_dir_l=None, mode='clam', num_classes=2):
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir_s = data_dir_s
		self.data_dir_l = data_dir_l
		self.mode = mode
		self.num_classes = num_classes
		# 不调用 Generic_MIL_Dataset.__init__，但 __getitem__ 会用到 H5 解析缓存
		self._h5_index_s = None
		self._h5_index_l = None
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)