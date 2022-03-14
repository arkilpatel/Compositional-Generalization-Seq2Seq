import os
import sys
import pandas as pd
import math
import logging
import pdb
import random
from time import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW
from gensim import models
from src.components.contextual_embeddings import BertEncoder, RobertaEncoder
from src.utils.sentence_processing import *
from src.utils.logger import print_log, store_results
from src.utils.helper import save_checkpoint
from src.utils.evaluate import cal_score, stack_to_string

import wandb

from collections import OrderedDict

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.scale = nn.Parameter(torch.ones(1)) # nn.Parameter causes the tensor to appear in the model.parameters()

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # max_len x 1
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # torch.arange(0, d_model, 2) gives 2i
		pe[:, 0::2] = torch.sin(position * div_term) # all alternate columns 0 onwards
		pe[:, 1::2] = torch.cos(position * div_term) # all alternate columns 1 onwards
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		'''
			Args:
				x (tensor): embeddings | size : [max_len x batch_size x d_model]
			Returns:
				z (tensor) : embeddings with positional encoding | size : [max_len x batch_size x d_model]
		'''
		
		x = x + self.scale * self.pe[:x.size(0), :]
		z = self.dropout(x)
		return z

class TransformerModel(nn.Module):
	def __init__(self, config, voc1, voc2, device, logger, EOS_tag = '</s>', SOS_tag = '<s>'):
		super(TransformerModel, self).__init__()
		self.config = config
		self.device = device
		self.voc1 = voc1
		self.voc2 = voc2
		self.EOS_tag = EOS_tag
		self.SOS_tag = SOS_tag
		self.EOS_token = voc2.get_id(EOS_tag)
		self.SOS_token = voc2.get_id(SOS_tag)
		self.logger = logger

		self.logger.debug('Initialising Embeddings...')

		if self.config.embedding == 'bert':
			config.d_model = 768
			self.embedding1 = BertEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
		elif self.config.embedding == 'roberta':
			config.d_model = 768
			self.embedding1 = RobertaEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
		elif self.config.embedding == 'word2vec':
			config.d_model = 300
			self.embedding1  = nn.Embedding.from_pretrained(torch.FloatTensor(self._form_embeddings(self.config.word2vec_bin)), 
								freeze = self.config.freeze_emb)
		else:
			self.embedding1  = nn.Embedding(self.voc1.nwords, self.config.d_model)
			nn.init.uniform_(self.embedding1.weight, -1 * self.config.init_range, self.config.init_range)

		self.pos_embedding1 = PositionalEncoding(self.config.d_model, self.config.dropout)

		self.embedding2  = nn.Embedding(self.voc2.nwords, self.config.d_model)
		nn.init.uniform_(self.embedding2.weight, -1 * self.config.init_range, self.config.init_range)
		
		self.pos_embedding2 = PositionalEncoding(self.config.d_model, self.config.dropout)

		self.logger.debug('Embeddings initialised...')
		self.logger.debug('Building Transformer Model...')

		self.transformer = nn.Transformer(d_model=self.config.d_model, nhead=self.config.heads, 
												num_encoder_layers=self.config.encoder_layers, num_decoder_layers=self.config.decoder_layers, 
												dim_feedforward=self.config.d_ff, dropout=self.config.dropout)
		
		self.fc_out = nn.Linear(self.config.d_model, self.voc2.nwords)

		self.logger.debug('Transformer Model Built...')

		if self.config.freeze_emb:
			for par in self.embedding1.parameters():
				par.requires_grad = False
		
		if self.config.freeze_emb2:
			for par in self.embedding2.parameters():
				par.requires_grad = False
		
		if self.config.freeze_transformer_encoder:
			for par in self.transformer.encoder.parameters():
				par.requires_grad = False

		if self.config.freeze_transformer_decoder:
			for par in self.transformer.decoder.parameters():
				par.requires_grad = False

		if self.config.freeze_fc:
			for par in self.fc_out.parameters():
				par.requires_grad = False

		self.src_mask = None
		self.tag_mask = None
		self.trg_mask = None
		self.memory_mask = None

		self.logger.debug('Initalizing Optimizer and Criterion...')

		self._initialize_optimizer()

		self.criterion = nn.CrossEntropyLoss() # nn.CrossEntropyLoss() does both F.log_softmax() and nn.NLLLoss() 

		self.logger.info('All Model Components Initialized...')

	def _form_embeddings(self, file_path):
		'''
			Args:
				file_path (string): path of file with word2vec weights
			Returns:
				weight_req (tensor) : embedding matrix | size : [voc1.nwords x d_model]
		'''

		weights_all = models.KeyedVectors.load_word2vec_format(file_path, limit=200000, binary=True)
		weight_req  = torch.randn(self.voc1.nwords, self.config.d_model)
		for key, value in self.voc1.id2w.items():
			if value in weights_all:
				weight_req[key] = torch.FloatTensor(weights_all[value])

		return weight_req

	def _initialize_optimizer(self):
		self.params = list(self.pos_embedding1.parameters()) + list(self.pos_embedding2.parameters())

		self.non_emb_params = list(self.pos_embedding1.parameters()) + list(self.pos_embedding2.parameters())

		if not self.config.freeze_emb:
			self.params = self.params + list(self.embedding1.parameters())
		
		if not self.config.freeze_emb2:
			self.params = self.params + list(self.embedding2.parameters())
			self.non_emb_params = self.non_emb_params + list(self.embedding2.parameters())
		
		if not self.config.freeze_transformer_encoder:
			self.params = self.params + list(self.transformer.encoder.parameters())
			self.non_emb_params = self.non_emb_params + list(self.transformer.encoder.parameters())

		if not self.config.freeze_transformer_decoder:
			self.params = self.params + list(self.transformer.decoder.parameters())
			self.non_emb_params = self.non_emb_params + list(self.transformer.decoder.parameters())

		if not self.config.freeze_fc:
			self.params = self.params + list(self.fc_out.parameters())
			self.non_emb_params = self.non_emb_params + list(self.fc_out.parameters())

		if not self.config.freeze_emb:
			if self.config.opt == 'adam':
				self.optimizer = optim.Adam(
					[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
					{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			elif self.config.opt == 'adamw':
				self.optimizer = optim.AdamW(
					[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
					{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			elif self.config.opt == 'adadelta':
				self.optimizer = optim.Adadelta(
					[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
					{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			elif self.config.opt == 'asgd':
				self.optimizer = optim.ASGD(
					[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
					{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			else:
				self.optimizer = optim.SGD(
					[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
					{"params": self.non_emb_params, "lr": self.config.lr}]
				)
		else:
			if self.config.opt == 'adam':
				self.optimizer = optim.Adam(
					[{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			elif self.config.opt == 'adamw':
				self.optimizer = optim.AdamW(
					[{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			elif self.config.opt == 'adadelta':
				self.optimizer = optim.Adadelta(
					[{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			elif self.config.opt == 'asgd':
				self.optimizer = optim.ASGD(
					[{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			else:
				self.optimizer = optim.SGD(
					[{"params": self.non_emb_params, "lr": self.config.lr}]
				)

	def generate_square_subsequent_mask(self, sz):
		'''
			Args:
				sz (integer): max_len of sequence in target without EOS i.e. (T-1)
			Returns:
				mask (tensor) : square mask | size : [T-1 x T-1]
		'''

		mask = torch.triu(torch.ones(sz, sz), 1)
		mask = mask.masked_fill(mask==1, float('-inf'))
		return mask

	def make_len_mask(self, inp):
		'''
			Args:
				inp (tensor): input indices | size : [S x BS]
			Returns:
				mask (tensor) : pad mask | size : [BS x S]
		'''

		mask = (inp == self.EOS_token).transpose(0, 1)
		return mask
		# return (inp == self.EOS_token).transpose(0, 1)

	def forward(self, ques, src, trg):
		'''
			Args:
				ques (list): raw source input | size : [BS]
				src (tensor): source indices | size : [S x BS]
				trg (tensor): target indices | size : [T x BS]
			Returns:
				output (tensor) : Network output | size : [T-1 x BS x voc2.nwords]
		'''

		if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
			self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

		# trg_mask when T-1 = 4: [When decoding for position i, only indexes with 0 in the ith row are attended over]
		# tensor([[0., -inf, -inf, -inf],
		# 		[0., 0., -inf, -inf],
		# 		[0., 0., 0., -inf],
		# 		[0., 0., 0., 0.],
		
		if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
			src, src_tokens = self.embedding1(ques)
			src = src.transpose(0,1)
			# src: Tensor [S x BS x d_model]
			src_pad_mask = self.make_len_mask(src_tokens.transpose(0,1))
			src = self.pos_embedding1(src)
		else:
			src_pad_mask = self.make_len_mask(src)
			src = self.embedding1(src)
			src = self.pos_embedding1(src)
		
		trg_pad_mask = self.make_len_mask(trg)
		trg = self.embedding2(trg)
		trg = self.pos_embedding2(trg)

		output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask, memory_mask=self.memory_mask,
										src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask)
		
		output = self.fc_out(output)

		return output

	def trainer(self, src, input_seq1, input_seq2, config, device=None ,logger=None):
		'''
			Args:
				src (list): raw source input | size : [BS]
				input_seq1 (tensor): source indices | size : [S x BS]
				input_seq2 (tensor): target indices | size : [T x BS]
			Returns:
				fin_loss (float) : Train Loss
		'''

		self.optimizer.zero_grad() # zero out gradients from previous backprop computations

		output = self.forward(src, input_seq1, input_seq2[:-1,:])
		# output: (T-1) x BS x voc2.nwords [T-1 because it predicts after start symbol]

		output_dim = output.shape[-1]
		
		self.loss = self.criterion(output.view(-1, output_dim), input_seq2[1:,:].view(-1))

		self.loss.backward()
		if self.config.max_grad_norm > 0:
			torch.nn.utils.clip_grad_norm_(self.params, self.config.max_grad_norm)
		self.optimizer.step()

		fin_loss = self.loss.item()

		return fin_loss

	def greedy_decode(self, src=None, input_seq1=None, input_seq2=None, input_len2 = None, validation=False):
		'''
			Args:
				src (list): raw source input | size : [BS]
				input_seq1 (tensor): source indices | size : [S x BS]
				input_seq2 (tensor): target indices | size : [T x BS]
				input_len2 (list): lengths of targets | size: [BS]
				validation (bool): whether validate
			Returns:
				if validation:
					validation loss (float): Validation loss
					decoded_words (list): predicted equations | size : [BS x target_len]
				else:
					decoded_words (list): predicted equations | size : [BS x target_len]
		'''

		with torch.no_grad():
			loss = 0.0

			if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
				src, _ = self.embedding1(src)
				src = src.transpose(0,1)
				# src: Tensor [S x BS x emb1_size]
				memory = self.transformer.encoder(self.pos_embedding1(src))
			else:
				memory = self.transformer.encoder(self.pos_embedding1(self.embedding1(input_seq1)))

			# memory: S x BS x d_model

			input_list = [[self.SOS_token for i in range(input_seq1.size(1))]]
			eos_list = torch.LongTensor([self.EOS_token for i in range(input_seq2.size(1))]).to(self.device) # BS

			decoded_words = [[] for i in range(input_seq1.size(1))]

			if validation:
				target_len = max(input_len2)
			else:
				target_len = self.config.max_length

			max_step = 1

			for step in range(target_len):
				decoder_input = torch.LongTensor(input_list).to(self.device) # seq_len x bs

				decoder_output = self.fc_out(self.transformer.decoder(self.pos_embedding2(self.embedding2(decoder_input)), memory)) # seq_len x bs x voc2.nwords
				
				if validation:
					loss += self.criterion(decoder_output[-1,:,:], input_seq2[step])
				else:
					if step < input_seq2.size(0):
						loss += self.criterion(decoder_output[-1,:,:], input_seq2[step])
					else:
						loss += self.criterion(decoder_output[-1,:,:], eos_list)

				out_tokens = decoder_output.argmax(2)[-1,:] # bs

				break_flag = 1

				for i in range(input_seq1.size(1)):
					if out_tokens[i].item() == self.EOS_token:
						continue
					break_flag = 0
					decoded_words[i].append(self.voc2.get_word(out_tokens[i].item()))

				if break_flag:
					if step > 1:
						max_step = step
					break
				
				input_list.append(out_tokens.detach().tolist())

			if validation:
				return loss/target_len, decoded_words
			else:
				return loss/max_step, decoded_words

def build_model(config, voc1, voc2, device, logger):
	'''
		Args:
			config (dict): command line arguments
			voc1 (object of class Voc1): vocabulary of source
			voc2 (object of class Voc2): vocabulary of target
			device (torch.device): GPU device
			logger (logger): logger variable to log messages
		Returns:
			model (object of class TransformerModel): model 
	'''

	model = TransformerModel(config, voc1, voc2, device, logger)
	model = model.to(device)

	return model

def train_model(model, train_dataloader, val_dataloader, test_dataloader, gen_dataloader, voc1, voc2, device, config, logger, min_train_loss=float('inf'), min_val_loss=float('inf'), min_test_loss=float('inf'),
				min_gen_loss=float('inf'), max_train_acc = 0.0, max_val_acc = 0.0, max_test_acc = 0.0, max_gen_acc = 0.0, best_epoch = 0):
	'''
		Args:
			model (object of class TransformerModel): model
			train_dataloader (object of class Dataloader): dataloader for train set
			val_dataloader (object of class Dataloader): dataloader for dev set
			voc1 (object of class Voc1): vocabulary of source
			voc2 (object of class Voc2): vocabulary of target
			device (torch.device): GPU device
			config (dict): command line arguments
			logger (logger): logger variable to log messages
			epoch_offset (int): How many epochs of training already done
			min_val_loss (float): minimum validation loss
			max_val_acc (float): maximum validation accuracy score
			min_train_loss (float): minimum train loss
			max_train_acc (float): maximum train accuracy
			best_epoch (int): epoch with highest validation accuracy
			writer (object of class SummaryWriter): writer for Tensorboard
		Returns:
			max_val_acc (float): maximum validation accuracy score
	'''
	
	estop_count=0
	
	for epoch in range(1, config.epochs + 1):
		od = OrderedDict()
		od['Epoch'] = epoch
		print_log(logger, od)

		batch_num = 1
		train_loss_epoch = 0.0
		train_acc_epoch = 0.0
		train_acc_epoch_cnt = 0.0
		train_acc_epoch_tot = 0.0

		start_time= time()
		total_batches = len(train_dataloader)

		for data in train_dataloader:
			src = data['src']

			sent1s = sents_to_idx(voc1, data['src'], config.max_length, flag=0)
			sent2s = sents_to_idx(voc2, data['trg'], config.max_length, flag=1)
			sent1_var, sent2_var, input_len1, input_len2  = process_batch(sent1s, sent2s, voc1, voc2, device)

			model.train()

			loss = model.trainer(src, sent1_var, sent2_var, config, device, logger)
			train_loss_epoch += loss

			wandb.log({"train loss per step": loss})

			if config.show_train_acc:
				model.eval()

				_, decoder_output = model.greedy_decode(src, sent1_var, sent2_var, input_len2, validation=True)
				temp_acc_cnt, temp_acc_tot, _ = cal_score(decoder_output, data['trg'])
				train_acc_epoch_cnt += temp_acc_cnt
				train_acc_epoch_tot += temp_acc_tot

			print("Completed {} / {}...".format(batch_num, total_batches), end = '\r', flush = True)
			batch_num+=1

		train_loss_epoch = train_loss_epoch / (len(train_dataloader) * config.batch_size)

		wandb.log({"train loss per epoch": train_loss_epoch})

		if config.show_train_acc:
			train_acc_epoch = train_acc_epoch_cnt/train_acc_epoch_tot
			wandb.log({"train accuracy": train_acc_epoch})
		else:
			train_acc_epoch = 0.0

		time_taken = (time() - start_time)/60.0

		logger.debug('Training for epoch {} completed...\nTime Taken: {}'.format(epoch, time_taken))
		
		if config.dev_set:
			logger.debug('Evaluating on Validation Set:')

		if config.dev_set and (config.dev_always or epoch >= config.epochs - (config.eval_last_n - 1)):
			val_loss_epoch, val_acc_epoch = run_validation(config=config, model=model, val_dataloader=val_dataloader, disp_tok='DEV', voc1=voc1, voc2=voc2, device=device, logger=logger, epoch_num = epoch, validation = True)
			wandb.log({"validation loss per epoch": val_loss_epoch})
			wandb.log({"validation accuracy": val_acc_epoch})
		else:
			val_loss_epoch = float('inf')
			val_acc_epoch = 0.0

		if config.test_set:
			logger.debug('Evaluating on Test Set:')
		
		if config.test_set and (config.test_always or epoch >= config.epochs - (config.eval_last_n - 1)):
			test_loss_epoch, test_acc_epoch = run_validation(config=config, model=model, val_dataloader=test_dataloader, disp_tok='TEST', voc1=voc1, voc2=voc2, device=device, logger=logger, epoch_num = epoch, validation = False)
			wandb.log({"test loss per epoch": test_loss_epoch})
			wandb.log({"test accuracy": test_acc_epoch})
		else:
			test_loss_epoch = float('inf')
			test_acc_epoch = 0.0

		if config.gen_set:
			logger.debug('Evaluating on Generalization Set:')

		if config.gen_set and (config.gen_always or epoch >= config.epochs - (config.eval_last_n - 1)):
			gen_loss_epoch, gen_acc_epoch = run_validation(config=config, model=model, val_dataloader=gen_dataloader, disp_tok='GEN', voc1=voc1, voc2=voc2, device=device, logger=logger, epoch_num = epoch, validation = False)
			wandb.log({"generalization loss per epoch": gen_loss_epoch})
			wandb.log({"generalization accuracy": gen_acc_epoch})
		else:
			gen_loss_epoch = float('inf')
			gen_acc_epoch = 0.0

		selector_flag = 0

		if train_loss_epoch < min_train_loss:
			min_train_loss = train_loss_epoch

		if train_acc_epoch > max_train_acc:
			max_train_acc = train_acc_epoch

		if val_loss_epoch < min_val_loss:
			min_val_loss = val_loss_epoch

		if val_acc_epoch > max_val_acc:
			max_val_acc = val_acc_epoch
			if config.model_selector_set == 'val':
				selector_flag = 1

		if test_loss_epoch < min_test_loss:
			min_test_loss = test_loss_epoch

		if test_acc_epoch > max_test_acc:
			max_test_acc = test_acc_epoch
			if config.model_selector_set == 'test':
				selector_flag = 1

		if gen_loss_epoch < min_gen_loss:
			min_gen_loss = gen_loss_epoch

		if gen_acc_epoch > max_gen_acc:
			max_gen_acc = gen_acc_epoch
			if config.model_selector_set == 'gen':
				selector_flag = 1
			
		if epoch == 1 or selector_flag == 1:
			best_epoch = epoch

			state = {
				'epoch' : epoch,
				'best_epoch': best_epoch,
				'model_state_dict': model.state_dict(),
				'voc1': model.voc1,
				'voc2': model.voc2,
				'optimizer_state_dict': model.optimizer.state_dict(),
				'train_loss_epoch' : train_loss_epoch,
				'min_train_loss' : min_train_loss,
				'train_acc_epoch' : train_acc_epoch,
				'max_train_acc' : max_train_acc,
				'val_loss_epoch' : val_loss_epoch,
				'min_val_loss' : min_val_loss,
				'val_acc_epoch' : val_acc_epoch,
				'max_val_acc' : max_val_acc,
				'test_loss_epoch' : test_loss_epoch,
				'min_test_loss' : min_test_loss,
				'test_acc_epoch' : test_acc_epoch,
				'max_test_acc' : max_test_acc,
				'gen_loss_epoch' : gen_loss_epoch,
				'min_gen_loss' : min_gen_loss,
				'gen_acc_epoch' : gen_acc_epoch,
				'max_gen_acc' : max_gen_acc,
			}

			if config.save_model:
				save_checkpoint(state, epoch, logger, config.model_path, config.ckpt)
			estop_count = 0
		else:
			estop_count += 1

		od = OrderedDict()
		od['Epoch'] = epoch
		od['best_epoch'] = best_epoch
		od['train_loss_epoch'] = train_loss_epoch
		od['min_train_loss'] = min_train_loss
		od['val_loss_epoch']= val_loss_epoch
		od['min_val_loss']= min_val_loss
		od['test_loss_epoch']= test_loss_epoch
		od['min_test_loss']= min_test_loss
		od['gen_loss_epoch']= gen_loss_epoch
		od['min_gen_loss']= min_gen_loss
		od['train_acc_epoch'] = train_acc_epoch
		od['max_train_acc'] = max_train_acc
		od['val_acc_epoch'] = val_acc_epoch
		od['max_val_acc'] = max_val_acc
		od['test_acc_epoch'] = test_acc_epoch
		od['max_test_acc'] = max_test_acc
		od['gen_acc_epoch'] = gen_acc_epoch
		od['max_gen_acc'] = max_gen_acc
		print_log(logger, od)

		if estop_count > config.early_stopping:
			logger.debug('Early Stopping at Epoch: {} after no improvement in {} epochs'.format(epoch, estop_count))
			break

	logger.info('Training Completed for {} epochs'.format(config.epochs))

	if config.results:
		store_results(config, max_train_acc, max_val_acc, max_test_acc, max_gen_acc, min_train_loss, min_val_loss, min_test_loss, min_gen_loss, best_epoch)
		logger.info('Scores saved at {}'.format(config.result_path))

	return max_val_acc

def run_validation(config, model, val_dataloader, disp_tok, voc1, voc2, device, logger, epoch_num, validation = True):
	'''
		Args:
			config (dict): command line arguments
			model (object of class TransformerModel): model
			val_dataloader (object of class Dataloader): dataloader for dev set
			voc1 (object of class Voc1): vocabulary of source
			voc2 (object of class Voc2): vocabulary of target
			device (torch.device): GPU device
			logger (logger): logger variable to log messages
			epoch_num (int): Ongoing epoch number
			validation (bool): whether validating
		Returns:
			if config.mode == 'test':
				max_test_acc (float): maximum test accuracy obtained
			else:
				val_loss_epoch (float): va;iadtion loss for this epoch
				val_acc (float): validation accuracy score for this epoch
	'''

	# if disp_tok == "GEN":
	# 	pdb.set_trace()

	batch_num = 1
	val_loss_epoch = 0.0
	val_acc_epoch = 0.0
	val_acc_epoch_cnt = 0.0
	val_acc_epoch_tot = 0.0

	model.eval() # Set specific layers such as dropout to evaluation mode

	refs= []
	hyps= []

	if config.mode == 'test':
		sources, gen_trgs, act_trgs, scores = [], [], [], []

	display_n = config.batch_size

	with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
		f_out.write('---------------------------------------\n')
		f_out.write('Set: '+ disp_tok +'\n')
		f_out.write('Epoch: ' + str(epoch_num) + '\n')
		f_out.write('---------------------------------------\n')
	total_batches = len(val_dataloader)
	for data in val_dataloader:
		sent1s = sents_to_idx(voc1, data['src'], config.max_length, flag = 0)
		sent2s = sents_to_idx(voc2, data['trg'], config.max_length, flag = 0)

		src = data['src']

		sent1_var, sent2_var, input_len1, input_len2 = process_batch(sent1s, sent2s, voc1, voc2, device)
		
		val_loss, decoder_output = model.greedy_decode(src, sent1_var, sent2_var, input_len2, validation)

		temp_acc_cnt, temp_acc_tot, disp_corr = cal_score(decoder_output, data['trg'])
		val_acc_epoch_cnt += temp_acc_cnt
		val_acc_epoch_tot += temp_acc_tot

		sent1s = idx_to_sents(voc1, sent1_var, no_eos= True)
		sent2s = idx_to_sents(voc2, sent2_var, no_eos= True)

		refs += [[' '.join(sent2s[i])] for i in range(sent2_var.size(1))]
		hyps += [' '.join(decoder_output[i]) for i in range(sent1_var.size(1))]

		if config.mode == 'test':
			sources+= data['src']
			gen_trgs += [' '.join(decoder_output[i]) for i in range(sent1_var.size(1))]
			act_trgs += [' '.join(sent2s[i]) for i in range(sent2_var.size(1))]
			scores   += [cal_score([decoder_output[i]], [data['trg'][i]])[0] for i in range(sent1_var.size(1))]

		with open(config.outputs_path + '/outputs.txt', 'a') as f_out:
			f_out.write('Batch: ' + str(batch_num) + '\n')
			f_out.write('---------------------------------------\n')
			for i in range(len(sent1s[:display_n])):
				try:
					f_out.write('Example: ' + str(i) + '\n')
					f_out.write('Source: ' + stack_to_string(sent1s[i]) + '\n')
					f_out.write('Target: ' + stack_to_string(sent2s[i]) + '\n')
					f_out.write('Generated: ' + stack_to_string(decoder_output[i]) + '\n')
					f_out.write('Result: ' + str(disp_corr[i]) + '\n')
					f_out.write('\n')
				except:
					logger.warning('Exception: Failed to generate')
					pdb.set_trace()
					break
			f_out.write('---------------------------------------\n')
			f_out.close()

		if batch_num % config.display_freq == 0:
			for i in range(len(sent1s[:display_n])):
				try:
					od = OrderedDict()
					logger.info('-------------------------------------')
					od['Source'] = ' '.join(sent1s[i])

					od['Target'] = ' '.join(sent2s[i])

					od['Generated'] = ' '.join(decoder_output[i])
					print_log(logger, od)
					logger.info('-------------------------------------')
				except:
					logger.warning('Exception: Failed to generate')
					pdb.set_trace()
					break

		val_loss_epoch += val_loss

		print("Completed {} / {}...".format(batch_num, total_batches), end = '\r', flush = True)
		batch_num += 1

	if config.mode == 'test':
		results_df = pd.DataFrame([sources, act_trgs, gen_trgs, scores]).transpose()
		results_df.columns = ['Source', 'Actual Target', 'Generated Target', 'Score']
		csv_file_path = os.path.join(config.outputs_path, config.dataset+'.csv')
		results_df.to_csv(csv_file_path, index = False)
		return sum(scores)/len(scores)

	val_acc_epoch = val_acc_epoch_cnt/val_acc_epoch_tot

	return val_loss_epoch/(len(val_dataloader) * config.batch_size), val_acc_epoch