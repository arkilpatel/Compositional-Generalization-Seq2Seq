import torch
import torch.nn as nn

import pdb

class CustomTransformer(nn.Module):
	def __init__(self, d_model1, d_model2, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation = "relu"):
		super(CustomTransformer, self).__init__()

		encoder_layer1 = nn.TransformerEncoderLayer(d_model1, nhead, dim_feedforward, dropout, activation)
		encoder_norm1 = nn.LayerNorm(d_model1)
		self.encoder1 = nn.TransformerEncoder(encoder_layer1, num_encoder_layers, encoder_norm1)

		encoder_layer2 = nn.TransformerEncoderLayer(d_model2, nhead, dim_feedforward, dropout, activation)
		encoder_norm2 = nn.LayerNorm(d_model2)
		self.encoder2 = nn.TransformerEncoder(encoder_layer2, num_encoder_layers, encoder_norm2)

		decoder_layer = nn.TransformerDecoderLayer(d_model1 + d_model2, nhead, dim_feedforward, dropout, activation)
		decoder_norm = nn.LayerNorm(d_model1 + d_model2)
		self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

		self._reset_parameters()

	def _reset_parameters(self):
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	def forward(self, src, tag, tgt, src_mask, tag_mask, tgt_mask, memory_mask, src_key_padding_mask, tag_key_padding_mask, 
				tgt_key_padding_mask, memory_key_padding_mask):

		memory1 = self.encoder1(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
		memory2 = self.encoder2(tag, mask=tag_mask, src_key_padding_mask=tag_key_padding_mask)

		memory = torch.cat((memory1, memory2), 2)

		output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, 
								memory_key_padding_mask=memory_key_padding_mask)

		return output