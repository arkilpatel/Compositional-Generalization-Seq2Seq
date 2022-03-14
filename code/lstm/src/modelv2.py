import os
from pickle import TRUE
import sys
import math
import logging
import pdb
import random
from time import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW
from gensim import models
from src.components.encoder import Encoder
from src.components.decoder import DecoderRNN
from src.components.attention import LuongAttnDecoderRNN
from src.components.contextual_embeddings import BertEncoder, RobertaEncoder
from src.utils.sentence_processing import *
from src.utils.logger import print_log, store_results
from src.utils.helper import save_checkpoint, bleu_scorer
from src.utils.evaluate import cal_score, stack_to_string
from src.confidence_estimation import *
from collections import OrderedDict

import wandb

class Seq2SeqModel(nn.Module):
	def __init__(self, config, voc1, voc2, device, logger, EOS_tag='</s>', SOS_tag='<s>'):
		super(Seq2SeqModel, self).__init__()

		self.config = config
		self.device = device
		self.voc1 = voc1
		self.voc2 = voc2
		self.EOS_tag = EOS_tag
		self.SOS_tag = SOS_tag
		self.EOS_token = voc2.get_id(EOS_tag)
		self.SOS_token = voc2.get_id(SOS_tag)
		self.logger = logger

		if self.config.embedding == 'bert':
			self.embedding1 = BertEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
		elif self.config.embedding == 'roberta':
			self.embedding1 = RobertaEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
		elif self.config.embedding == 'word2vec':
			self.config.emb1_size = 300
			self.embedding1 = nn.Embedding.from_pretrained(torch.FloatTensor(self._form_embeddings(self.config.word2vec_bin)), freeze = self.config.freeze_emb)
		else:
			self.embedding1  = nn.Embedding(self.voc1.nwords, self.config.emb1_size)
			nn.init.uniform_(self.embedding1.weight, -1 * self.config.init_range, self.config.init_range)

		self.logger.debug('Building Encoders...')
		self.encoder = Encoder(
			self.config.hidden_size,
			self.config.emb1_size,
			self.config.cell_type,
			self.config.depth,
			self.config.dropout,
			self.config.bidirectional
		)

		self.logger.debug('Encoders Built...')

		if self.config.use_attn:
			self.decoder    = LuongAttnDecoderRNN(self.config,
												  self.voc2).to(device)
		else:
			self.decoder    = DecoderRNN(self.config,
										 self.voc2,
										 self.config.cell_type,
										 self.config.hidden_size,
										 self.voc2.nwords,
										 self.config.depth,
										 self.config.dropout).to(device)

		self.logger.debug('Decoder RNN Built...')

		if self.config.freeze_emb:
			for par in self.embedding1.parameters():
				par.requires_grad = False
		
		if self.config.freeze_emb2:
			for par in self.decoder.embedding.parameters():
				par.requires_grad = False
			for par in self.decoder.embedding_dropout.parameters():
				par.requires_grad = False
		
		if self.config.freeze_lstm_encoder:
			for par in self.encoder.parameters():
				par.requires_grad = False

		if self.config.freeze_lstm_decoder:
			for par in self.decoder.parameters():
				if par not in self.decoder.embedding.parameters() and par not in self.decoder.embedding_dropout.parameters() and par not in self.decoder.out.parameters():
					par.requires_grad = False

		if self.config.freeze_fc:
			for par in self.decoder.out.parameters():
				par.requires_grad = False

		self.logger.debug('Initalizing Optimizer and Criterion...')
		self._initialize_optimizer()

		# nn.CrossEntropyLoss() does both F.log_softmax() and nn.NLLLoss() 
		self.criterion = nn.NLLLoss() 

		self.logger.info('All Model Components Initialized...')

	def _form_embeddings(self, file_path):
		weights_all = models.KeyedVectors.load_word2vec_format(file_path, limit=200000, binary=True)
		weight_req  = torch.randn(self.voc1.nwords, self.config.emb1_size)
		for key, value in self.voc1.id2w.items():
			if value in weights_all:
				weight_req[key] = torch.FloatTensor(weights_all[value])

		return weight_req	

	def _initialize_optimizer(self):
		self.params =   list()
		self.non_emb_params = list()

		if not self.config.freeze_emb:
			self.params = self.params + list(self.embedding1.parameters())

		if not self.config.freeze_emb2:
			self.params = self.params + list(self.decoder.embedding.parameters()) + list(self.decoder.embedding_dropout.parameters())
			self.non_emb_params = self.non_emb_params + list(self.decoder.embedding.parameters())
		
		if not self.config.freeze_lstm_encoder:
			self.params = self.params + list(self.encoder.parameters())
			self.non_emb_params = self.non_emb_params + list(self.encoder.parameters())
		
		if not self.config.freeze_lstm_decoder:
			if self.config.use_attn:
				decoder_only_params = list(self.decoder.rnn.parameters()) + list(self.decoder.concat.parameters()) + list(self.decoder.attn.parameters())
			else:
				decoder_only_params = list(self.decoder.rnn.parameters())
			self.params = self.params + decoder_only_params
			self.non_emb_params = self.non_emb_params + decoder_only_params

		if not self.config.freeze_fc:
			self.params = self.params + list(self.decoder.out.parameters())
			self.non_emb_params = self.non_emb_params + list(self.decoder.out.parameters())

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

	def forward(self, input_seq1, input_seq2, input_len1, input_len2):
		'''
			Args:
				input_seq1 (tensor): values are word indexes | size : [max_len x batch_size]
				input_len1 (tensor): Length of each sequence in input_len1 | size : [batch_size]
				input_seq2 (tensor): values are word indexes | size : [max_len x batch_size]
				input_len2 (tensor): Length of each sequence in input_len2 | size : [batch_size]
			Returns:
				out (tensor) : Probabilities of each output label for each point | size : [batch_size x num_labels]
		'''

	def trainer(self, src, input_seq1, input_seq2, input_len1, input_len2, config, device=None ,logger=None):
		'''
			Args:
				src (list): input examples as is (i.e. not indexed) | size : [batch_size]
			Returns:
				
		'''
		self.optimizer.zero_grad()

		if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
			input_seq1, input_len1 = self.embedding1(src)
			input_seq1 = input_seq1.transpose(0,1)
			# input_seq1: Tensor [max_len x BS x emb1_size]
			# input_len1: List [BS]
			sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq1, input_len1, self.device)
			# sorted_seqs: Tensor [max_len x BS x emb1_size]
			# input_len1: List [BS]
			# orig_idx: Tensor [BS]
		else:
			sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq1, input_len1, self.device)
			sorted_seqs = self.embedding1(sorted_seqs)

		encoder_outputs, encoder_hidden = self.encoder(sorted_seqs, sorted_len, orig_idx, self.device)
		
		self.loss = 0

		decoder_input = torch.tensor([self.SOS_token for i in range(input_seq1.size(1))], device = self.device)

		if config.cell_type == 'lstm':
			decoder_hidden = (encoder_hidden[0][:self.decoder.nlayers], encoder_hidden[1][:self.decoder.nlayers])
		else:
			decoder_hidden = encoder_hidden[:self.decoder.nlayers]

		use_teacher_forcing = True if random.random() < self.config.teacher_forcing_ratio else False
		target_len = max(input_len2)

		if use_teacher_forcing:
			for step in range(target_len):
				if self.config.use_attn:
					decoder_output, decoder_hidden, decoder_attention, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
				else:
					decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
				self.loss += self.criterion(decoder_output, input_seq2[step])
				decoder_input = input_seq2[step]
		else:
			for step in range(target_len):
				if self.config.use_attn:
					decoder_output, decoder_hidden, decoder_attention, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
				else:
					decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
				
				topv, topi = decoder_output.topk(1)
				self.loss += self.criterion(decoder_output, input_seq2[step])
				decoder_input = topi.squeeze().detach() 

		self.loss.backward()
		if self.config.max_grad_norm > 0:
			torch.nn.utils.clip_grad_norm_(self.params, self.config.max_grad_norm)
		self.optimizer.step()

		return self.loss.item()/target_len

	def greedy_decode(self, src, input_seq1=None, input_seq2=None, input_len1=None, input_len2=None, validation=False, return_probs = False):
		with torch.no_grad():
			if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
				input_seq1, input_len1 = self.embedding1(src)
				input_seq1 = input_seq1.transpose(0,1)
				sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq1, input_len1, self.device)
			else:
				sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq1, input_len1, self.device)
				sorted_seqs = self.embedding1(sorted_seqs)

			encoder_outputs, encoder_hidden = self.encoder(sorted_seqs, sorted_len, orig_idx, self.device)

			loss = 0.0
			decoder_input = torch.tensor([self.SOS_token for i in range(input_seq1.size(1))], device=self.device)
			eos_list = torch.LongTensor([self.EOS_token for i in range(input_seq2.size(1))]).to(self.device) # BS

			if self.config.cell_type == 'lstm':
				decoder_hidden = (encoder_hidden[0][:self.decoder.nlayers], encoder_hidden[1][:self.decoder.nlayers])
			else:
				decoder_hidden = encoder_hidden[:self.decoder.nlayers]

			decoded_words = [[] for i in range(input_seq1.size(1))]
			decoded_probs = [[] for i in range(input_seq1.size(1))]
			decoder_attentions = []

			if validation:
				target_len = max(input_len2)
			else:
				target_len = self.config.max_length

			max_step = 1

			for step in range(target_len):
				if self.config.use_attn:
					decoder_output, decoder_hidden, decoder_attention, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
					decoder_attentions.append(decoder_attention)
				else:
					decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

				if validation:
					loss += self.criterion(decoder_output, input_seq2[step])
				else:
					if step < input_seq2.size(0):
						loss += self.criterion(decoder_output, input_seq2[step])
					else:
						loss += self.criterion(decoder_output, eos_list)
				
				topv, topi = decoder_output.topk(1)

				break_flag = 1

				for i in range(input_seq1.size(1)):
					if topi[i].item() == self.EOS_token:
						continue
					break_flag = 0
					decoded_words[i].append(self.voc2.get_word(topi[i].item()))
					decoded_probs[i].append(topv[i].item())

				if break_flag:
					if step > 1:
						max_step = step
					break
				
				decoder_input = topi.squeeze().detach()

			if validation:
				if self.config.use_attn:
					return loss/target_len, decoded_words, decoder_attentions[:step + 1]
				else:
					return loss/target_len, decoded_words, None
			else:
				if return_probs:
					return decoded_words, decoded_probs, None

				return loss/max_step, decoded_words, None

	def obtain_hidden(self, config, ques, input_seq1=None, input_seq2=None, input_len1=None, input_len2=None):
		with torch.no_grad():
			if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
				input_seq1, input_len1 = self.embedding1(ques)
				input_seq1 = input_seq1.transpose(0,1)
				sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq1, input_len1, self.device)
			else:
				sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq1, input_len1, self.device)
				sorted_seqs = self.embedding1(sorted_seqs)

			encoder_outputs, encoder_hidden = self.encoder(sorted_seqs, sorted_len, orig_idx, self.device)

			loss =0.0
			decoder_input = torch.tensor([self.SOS_token for i in range(input_seq1.size(1))], device=self.device)

			if self.config.cell_type == 'lstm':
				decoder_hidden = (encoder_hidden[0][:self.decoder.nlayers], encoder_hidden[1][:self.decoder.nlayers])
			else:
				decoder_hidden = encoder_hidden[:self.decoder.nlayers]

			decoded_words = [[] for i in range(input_seq1.size(1))]
			decoder_attentions = []

			hiddens = []

			target_len = max(input_len2)

			for step in range(target_len):
				if self.config.use_attn:
					decoder_output, decoder_hidden, decoder_attention, hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
					decoder_attentions.append(decoder_attention)
				else:
					decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

				topv, topi = decoder_output.topk(1)
				for i in range(input_seq1.size(1)):
					if topi[i].item() == self.EOS_token:
						continue
					decoded_words[i].append(self.voc2.get_word(topi[i].item()))
					hiddens.append([self.voc2.get_word(topi[i].item()), hidden[i]])
				decoder_input = topi.squeeze().detach()

			return hiddens, decoded_words

def build_model(config, voc1, voc2, device, logger):
	'''
		Add Docstring
	'''
	model = Seq2SeqModel(config, voc1, voc2, device, logger)
	model = model.to(device)

	return model

def train_model(model, train_dataloader, val_dataloader, test_dataloader, gen_dataloader, voc1, voc2, device, config, logger, min_train_loss=float('inf'), min_val_loss=float('inf'), min_test_loss=float('inf'), 
				min_gen_loss=float('inf'), max_train_acc = 0.0, max_val_acc = 0.0, max_test_acc = 0.0, max_gen_acc = 0.0, best_epoch = 0):
	'''
		Add Docstring
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

			sent1s = sents_to_idx(voc1, data['src'], config.max_length)
			sent2s = sents_to_idx(voc2, data['trg'], config.max_length)
			sent1_var, sent2_var, input_len1, input_len2  = process_batch(sent1s, sent2s, voc1, voc2, device)

			model.train()

			loss = model.trainer(src, sent1_var, sent2_var, input_len1, input_len2, config, device, logger)
			train_loss_epoch += loss

			wandb.log({"train loss per step": loss})

			if config.show_train_acc:
				model.eval()

				_, decoder_output, _ = model.greedy_decode(src, sent1_var, sent2_var, input_len1, input_len2, validation=True)
				temp_acc_cnt, temp_acc_tot, _ = cal_score(decoder_output, data['trg'])
				train_acc_epoch_cnt += temp_acc_cnt
				train_acc_epoch_tot += temp_acc_tot

			batch_num+=1
			print("Completed {} / {}...".format(batch_num, total_batches), end = '\r', flush = True)

		train_loss_epoch = train_loss_epoch/len(train_dataloader)

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
			val_loss_epoch, val_acc_epoch = run_validation(config=config, model=model, dataloader=val_dataloader, disp_tok='DEV', voc1=voc1, voc2=voc2, device=device, logger=logger, epoch_num = epoch, validation = True)
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
			estop_count+=1

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

def run_validation(config, model, dataloader, disp_tok, voc1, voc2, device, logger, epoch_num, validation = False):
	batch_num = 1
	val_loss_epoch = 0.0
	val_acc_epoch = 0.0
	val_acc_epoch_cnt = 0.0
	val_acc_epoch_tot = 0.0

	model.eval()

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
	total_batches = len(dataloader)
	for data in dataloader:
		sent1s = sents_to_idx(voc1, data['src'], config.max_length)
		sent2s = sents_to_idx(voc2, data['trg'], config.max_length)

		src = data['src']

		sent1_var, sent2_var, input_len1, input_len2 = process_batch(sent1s, sent2s, voc1, voc2, device)

		val_loss, decoder_output, _ = model.greedy_decode(src, sent1_var, sent2_var, input_len1, input_len2, validation)

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

		if batch_num % config.display_freq ==0:
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
		batch_num +=1
		print("Completed {} / {}...".format(batch_num, total_batches), end = '\r', flush = True)

	if config.mode == 'test':
		results_df = pd.DataFrame([sources, act_trgs, gen_trgs, scores]).transpose()
		results_df.columns = ['Source', 'Actual Target', 'Generated Target', 'Score']
		csv_file_path = os.path.join(config.outputs_path, config.dataset+'.csv')
		results_df.to_csv(csv_file_path, index = False)
		return sum(scores)/len(scores)

	val_acc_epoch = val_acc_epoch_cnt/val_acc_epoch_tot

	return val_loss_epoch/(len(dataloader) * config.batch_size), val_acc_epoch

def estimate_confidence(config, model, dataloader, logger):
	
	questions	= []
	act_eqns 	= []
	gen_eqns	= []
	scores		= []
	confs		= []
	batch_num = 0
	
	#Load training data (Will be useful for similarity based methods)
	train_df 	= pd.read_csv(os.path.join('data',config.dataset,'train.csv'))
	train_ques	= train_df['Question'].values 
	
	total_batches = len(dataloader)
	logger.info("Beginning estimating confidence based on {} criteria".format(config.conf))
	start = time()
	for data in dataloader:
		ques, eqn, nums, ans = data['ques'], data['eqn'], data['nums'], data['ans']
		
		if config.conf == 'posterior':
			decoded_words, confidence = posterior_based_conf(ques, model)
		elif config.conf == 'similarity':
			decoded_words, confidence = similarity_based_conf(ques, train_ques, model, sim_criteria= config.sim_criteria)
		else:
			#TODO: Implement other methods
			raise ValueError("Other confidence methods not implemented yet. Use -conf posterior")
		
		if not config.adv:
			correct_or_not = [cal_score([decoded_words[i]], [nums[i]], [ans[i]])[0] for i in range(len(decoded_words))]
		else:
			correct_or_not = [-1 for i in range(len(decoded_words))]

		gen_eqn = [' '.join(words) for words in decoded_words]
		
		questions 	+= ques
		act_eqns	+= eqn
		gen_eqns	+= gen_eqn
		scores		+= correct_or_not
		confs		+= list(confidence)
		batch_num	+= 1
		print("Completed {} / {}...".format(batch_num, total_batches), end = '\r', flush = True)

	results_df = pd.DataFrame([questions, act_eqns, gen_eqns, scores, confs]).transpose()
	results_df.columns = ['Question', 'Actual Equation', 'Generated Equation', 'Score', 'Confidence']
	if config.conf != 'similarity':
		csv_file_path = os.path.join('ConfidenceEstimates',config.dataset + '_' + config.run_name + '_' + config.conf + '.csv')
	else:
		csv_file_path = os.path.join('ConfidenceEstimates',config.dataset + '_' + config.run_name + '_' + config.conf + '_' + config.sim_criteria + '.csv')
	results_df.to_csv(csv_file_path)
	logger.info("Done in {} seconds".format(time() - start))

def get_hiddens(config, model, val_dataloader, voc1, voc2, device):
	batch_num =1
	
	model.eval()

	hiddens = []
	operands = []

	for data in val_dataloader:
		if len(data['ques']) == config.batch_size:
			sent1s = sents_to_idx(voc1, data['ques'], config.max_length)
			sent2s = sents_to_idx(voc2, data['eqn'], config.max_length)
			nums = data['nums']
			ans = data['ans']

			ques = data['ques']

			sent1_var, sent2_var, input_len1, input_len2 = process_batch(sent1s, sent2s, voc1, voc2, device)

			hidden, decoder_output = model.obtain_hidden(config, ques, sent1_var, sent2_var, input_len1, input_len2)

			infix = get_infix_eq(decoder_output, nums)[0] # WORKS ONLY FOR BATCH SIZE 1
			words = infix.split()

			type_rep = []
			operand_types = []

			for w in range(len(words)):
				if words[w] == '/':
					if words[w-1][0] == 'n':
						operand_types.append(['dividend', words[w-1]])
					if words[w+1][0] == 'n':
						operand_types.append(['divisor', words[w+1]])
				elif words[w] == '-':
					if words[w-1][0] == 'n':
						operand_types.append(['minuend', words[w-1]])
					if words[w+1][0] == 'n':
						operand_types.append(['subtrahend', words[w+1]])

			for z in range(len(operand_types)):
				entity = operand_types[z][1]
				for y in range(len(hidden)):
					if hidden[y][0] == entity:
						type_rep.append([operand_types[z][0], hidden[y][1]])

			hiddens = hiddens + hidden
			operands = operands + type_rep

	return hiddens, operands


