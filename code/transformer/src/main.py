# python -m src.main -mode train -project_name test_runs -model_selector_set val -pretrained_model_name none -finetune_data_voc none -dev_set -no-test_set -no-gen_set -dataset add_jump_100_prims_controlled_10_prims_test -dev_always -no-test_always -no-gen_always -epochs 150 -save_model -no-show_train_acc -embedding random -no-freeze_emb -no-freeze_emb2 -no-freeze_transformer_encoder -no-freeze_transformer_decoder -no-freeze_fc -d_model 64 -d_ff 512 -decoder_layers 3 -encoder_layers 3 -heads 2 -batch_size 64 -lr 0.0005 -emb_lr 0.0005 -dropout 0.1 -run_name RUN-train_try -gpu 1
# python -m src.main -mode test -project_name test_runs -pretrained_model_name RUN-train_try -finetune_data_voc none -no-dev_set -no-test_set -gen_set -dataset add_jump_100_prims_controlled_10_prims_test -batch_size 256 -run_name RUN-test_try -gpu 1
import os
import sys
import math
import logging
import pdb
import random
import numpy as np
from attrdict import AttrDict
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
try:
	import cPickle as pickle
except ImportError:
	import pickle

import wandb

from src.args import build_parser
from src.utils.helper import *
from src.utils.logger import get_logger, print_log
from src.dataloader import TextDataset
from src.model import build_model, train_model, run_validation

global log_folder
global model_folder
global result_folder
global data_path

log_folder = 'logs'
model_folder = 'models'
outputs_folder = 'outputs'
result_folder = './out/'
data_path = './data/'

def load_data(config, logger):
	'''
		Loads the data from the datapath in torch dataset form

		Args:
			config (dict) : configuration/args
			logger (logger) : logger object for logging

		Returns:
			dataloader(s) 
	'''
	
	if config.mode == 'train':
		logger.debug('Loading Training Data...')

		'''Load Datasets'''
		train_set = TextDataset(data_path=data_path, dataset=config.dataset, datatype='train', max_length=config.max_length, 
								is_debug=config.debug, to_sort=True)
		train_dataloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=5)
		train_size = len(train_dataloader) * config.batch_size
		if config.dev_set:
			val_set = TextDataset(data_path=data_path, dataset=config.dataset, datatype='dev', max_length=config.max_length, 
									is_debug=config.debug, to_sort=True)
			val_dataloader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=5)
			val_size = len(val_dataloader)* config.batch_size
		else:
			val_dataloader = None
			val_size = 0
		if config.test_set:
			test_set = TextDataset(data_path=data_path, dataset=config.dataset, datatype='test', max_length=config.max_length, 
									is_debug=config.debug, to_sort=True)
			test_dataloader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=5)
			test_size = len(test_dataloader)* config.batch_size
		else:
			test_dataloader = None
			test_size = 0
		if config.gen_set:
			gen_set = TextDataset(data_path=data_path, dataset=config.dataset, datatype='gen', max_length=config.max_length, 
									is_debug=config.debug, to_sort=True)
			gen_dataloader = DataLoader(gen_set, batch_size=config.batch_size, shuffle=False, num_workers=5)
			gen_size = len(gen_dataloader)* config.batch_size
		else:
			gen_dataloader = None
			gen_size = 0

		finetune_dataloader = None
		if config.finetune_data_voc != 'none':
			finetune_set = TextDataset(data_path=data_path, dataset=config.finetune_data_voc, datatype='train', max_length=config.max_length, 
								is_debug=config.debug, to_sort=True)
			finetune_dataloader = DataLoader(finetune_set, batch_size=config.batch_size, shuffle=True, num_workers=5)
		
		msg = 'All Data Loaded:\nTrain Size: {}\nVal Size: {}\nTest Size: {}\nGen Size: {}'.format(train_size, val_size, test_size, gen_size)
		logger.info(msg)

		return train_dataloader, val_dataloader, test_dataloader, gen_dataloader, finetune_dataloader

	elif config.mode == 'test':
		logger.debug('Loading Test Data...')

		gen_set = TextDataset(data_path=data_path, dataset=config.dataset, datatype='gen', max_length=config.max_length, 
								is_debug=config.debug, to_sort=True)
		gen_dataloader = DataLoader(
			gen_set, batch_size=config.batch_size, shuffle=False, num_workers=5)

		logger.info('Test Data Loaded...')
		return gen_dataloader

	else:
		logger.critical('Invalid Mode Specified')
		raise Exception('{} is not a valid mode'.format(config.mode))


def main():
	'''read arguments'''
	parser = build_parser()
	args = parser.parse_args()

	config = args
	mode = config.mode
	if mode == 'train':
		is_train = True
	else:
		is_train = False

	''' Set seed for reproducibility'''
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	random.seed(config.seed)

	'''GPU initialization'''
	device = gpu_init_pytorch(config.gpu)

	'''Run Config files/paths'''
	run_name = config.run_name
	config.log_path = os.path.join(log_folder, run_name)
	config.model_path = os.path.join(model_folder, run_name)
	config.outputs_path = os.path.join(outputs_folder, run_name)

	if config.mode == 'test' and config.pretrained_model_name == 'none':
		config.pretrained_model_name == run_name

	wandb.init(project=config.project_name, entity="arkil")
	wandb.init(config={"lr": 0.1})
	wandb.config.epochs = 4
	wandb.config.update(args, allow_val_change=True) # adds all of the arguments as config variables

	vocab1_path = os.path.join(config.model_path, 'vocab1.p')
	vocab2_path = os.path.join(config.model_path, 'vocab2.p')
	config_file = os.path.join(config.model_path, 'config.p')
	log_file = os.path.join(config.log_path, 'log.txt')

	if config.results:
		config.result_path = os.path.join(result_folder, 'val_results_{}.json'.format(config.dataset))

	if is_train:
		create_save_directories(config.log_path)
		create_save_directories(config.model_path)
		create_save_directories(config.outputs_path)
	else:
		create_save_directories(config.log_path)
		create_save_directories(config.result_path)

	logger = get_logger(run_name, log_file, logging.DEBUG)

	logger.debug('Created Relevant Directories')
	logger.info('Experiment Name: {}'.format(config.run_name))

	'''Read Files and create/load Vocab'''
	if is_train:
		train_dataloader, val_dataloader, test_dataloader, gen_dataloader, finetune_dataloader = load_data(config, logger)

		if config.pretrained_model_name == "none":
			logger.debug('Creating Vocab...')

			voc1 = Voc()
			voc1.create_vocab_dict(config, train_dataloader, 'src')

			voc2 = Voc()
			voc2.create_vocab_dict(config, train_dataloader, 'trg')

			if val_dataloader is not None:
				voc2.add_to_vocab_dict(config, val_dataloader, 'trg')
			if test_dataloader is not None:
				voc2.add_to_vocab_dict(config, test_dataloader, 'trg')
			if gen_dataloader is not None:
				voc2.add_to_vocab_dict(config, gen_dataloader, 'trg')

			if finetune_dataloader is not None:
				voc1.add_to_vocab_dict(config, finetune_dataloader, 'src')
				voc2.add_to_vocab_dict(config, finetune_dataloader, 'trg')

			logger.info('Input Vocab Created with number of words : {}'.format(voc1.nwords))
			logger.info('Output Vocab Created with number of words : {}'.format(voc2.nwords))

			with open(vocab1_path, 'wb') as f:
				pickle.dump(voc1, f, protocol=pickle.HIGHEST_PROTOCOL)
			with open(vocab2_path, 'wb') as f:
				pickle.dump(voc2, f, protocol=pickle.HIGHEST_PROTOCOL)

			logger.info('Vocab saved at {}'.format(vocab1_path))
		else:
			logger.info('Loading Vocab File...')

			pretrained_model_path = os.path.join(model_folder, config.pretrained_model_name)
			vocab1_path = os.path.join(pretrained_model_path, 'vocab1.p')
			vocab2_path = os.path.join(pretrained_model_path, 'vocab2.p')

			with open(vocab1_path, 'rb') as f:
				voc1 = pickle.load(f)
			with open(vocab2_path, 'rb') as f:
				voc2 = pickle.load(f)

			logger.info('Vocab Files loaded from {}\nNumber of Words: {}'.format(vocab1_path, voc1.nwords))

	else:
		gen_dataloader = load_data(config, logger)

		create_save_directories(config.model_path)
		create_save_directories(config.outputs_path)
		logger.info('Loading Vocab File...')

		pretrained_model_path = os.path.join(model_folder, config.pretrained_model_name)
		vocab1_path = os.path.join(pretrained_model_path, 'vocab1.p')
		vocab2_path = os.path.join(pretrained_model_path, 'vocab2.p')
		config_file = os.path.join(pretrained_model_path, 'config.p')

		with open(vocab1_path, 'rb') as f:
			voc1 = pickle.load(f)
		with open(vocab2_path, 'rb') as f:
			voc2 = pickle.load(f)

		logger.info('Vocab Files loaded from {}\nNumber of words in voc1: {}\nNumber of words in voc2: {}'.format(vocab1_path, voc1.nwords, voc2.nwords))

	if is_train:
		model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger)

		logger.info('Initialized Model')

		min_val_loss = torch.tensor(float('inf')).item()
		min_test_loss = torch.tensor(float('inf')).item()
		min_gen_loss = torch.tensor(float('inf')).item()
		min_train_loss = torch.tensor(float('inf')).item()
		max_val_acc = 0.0
		max_test_acc = 0.0
		max_gen_acc = 0.0
		max_train_acc = 0.0
		best_epoch = 0

		if config.pretrained_model_name != "none":
			pretrained_model_path = os.path.join(model_folder, config.pretrained_model_name)
			pretrained_checkpoint = get_latest_checkpoint(pretrained_model_path, logger)
			_, _, _, _, _, _, _, _, _, _, _, _ = load_checkpoint(model, config.mode, pretrained_checkpoint, logger, device, pretrained = True)

		with open(config_file, 'wb') as f:
			pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

		logger.debug('Config File Saved')

		logger.info('Starting Training Procedure')
		train_model(model, train_dataloader, val_dataloader, test_dataloader, gen_dataloader, voc1, voc2, device, config, logger, 
					min_train_loss, min_val_loss, min_test_loss, min_gen_loss, max_train_acc, max_val_acc, max_test_acc, max_gen_acc, best_epoch)

	else:
		gpu = config.gpu
		mode = config.mode
		dataset = config.dataset
		batch_size = config.batch_size
		with open(config_file, 'rb') as f:
			config = AttrDict(pickle.load(f))
			config.gpu = gpu
			config.mode = mode
			config.dataset = dataset
			config.batch_size = batch_size

		model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger)

		logger.info('Initialized Model')

		pretrained_checkpoint = get_latest_checkpoint(pretrained_model_path, logger)

		epoch_offset, min_train_loss, min_val_loss, min_test_loss, min_gen_loss, max_train_acc, max_val_acc, max_test_acc, max_gen_acc, best_epoch, voc1, voc2 = load_checkpoint(model, config.mode, pretrained_checkpoint, logger, device, pretrained = True)

		logger.info('Prediction from')
		od = OrderedDict()
		od['epoch'] = epoch_offset
		od['min_train_loss'] = min_train_loss
		od['min_val_loss'] = min_val_loss
		od['min_test_loss'] = min_test_loss
		od['min_gen_loss'] = min_gen_loss
		od['max_train_acc'] = max_train_acc
		od['max_val_acc'] = max_val_acc
		od['max_test_acc'] = max_test_acc
		od['max_gen_acc'] = max_gen_acc
		od['best_epoch'] = best_epoch
		print_log(logger, od)

		gen_acc_epoch = run_validation(config=config, model=model, val_dataloader=gen_dataloader, disp_tok='GEN', voc1=voc1, voc2=voc2, device=device, logger=logger, epoch_num = 0)
		logger.info('Accuracy: {}'.format(gen_acc_epoch))


if __name__ == '__main__':
	main()


''' Just docstring format '''
# class Vehicles(object):
# 	'''
# 	The Vehicle object contains a lot of vehicles

# 	Args:
# 		arg (str): The arg is used for...
# 		*args: The variable arguments are used for...
# 		**kwargs: The keyword arguments are used for...

# 	Attributes:
# 		arg (str): This is where we store arg,
# 	'''
# 	def __init__(self, arg, *args, **kwargs):
# 		self.arg = arg

# 	def cars(self, distance,destination):
# 		'''We can't travel distance in vehicles without fuels, so here is the fuels

# 		Args:
# 			distance (int): The amount of distance traveled
# 			destination (bool): Should the fuels refilled to cover the distance?

# 		Raises:
# 			RuntimeError: Out of fuel

# 		Returns:
# 			cars: A car mileage
# 		'''
# 		pass