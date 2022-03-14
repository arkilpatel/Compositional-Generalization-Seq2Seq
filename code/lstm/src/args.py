import argparse

### Add Early Stopping ###

def build_parser():
	# Data loading parameters
	parser = argparse.ArgumentParser(description='Run Single sequence model')

	# Mode specifications
	parser.add_argument('-mode', type=str, default='train', choices=['train', 'test', 'conf'], help='Modes: train, test, conf')
	parser.add_argument('-debug', dest='debug', action='store_true', help='Operate in debug mode')
	parser.add_argument('-no-debug', dest='debug', action='store_false', help='Operate in normal mode')
	parser.set_defaults(debug=False)
	parser.add_argument('-dev_set', dest='dev_set', action='store_true', help='Evaluate on Dev set')
	parser.add_argument('-no-dev_set', dest='dev_set', action='store_false', help='Don\'t Evaluate on Dev set')
	parser.set_defaults(dev_set=False)
	parser.add_argument('-test_set', dest='test_set', action='store_true', help='Evaluate on Test set')
	parser.add_argument('-no-test_set', dest='test_set', action='store_false', help='Don\'t Evaluate on Test set')
	parser.set_defaults(test_set=False)
	parser.add_argument('-gen_set', dest='gen_set', action='store_true', help='Evaluate on Generalization set')
	parser.add_argument('-no-gen_set', dest='gen_set', action='store_false', help='Don\'t Evaluate on Generalization set')
	parser.set_defaults(gen_set=True)
	parser.add_argument('-gen_always', dest='gen_always', action='store_true', help='Evaluate on Gen set at each epoch')
	parser.add_argument('-no-gen_always', dest='gen_always', action='store_false', help='Evaluate on Gen set in the end')
	parser.set_defaults(gen_always=True)
	parser.add_argument('-dev_always', dest='dev_always', action='store_true', help='Evaluate on Dev set at each epoch')
	parser.add_argument('-no-dev_always', dest='dev_always', action='store_false', help='Evaluate on Dev set in the end')
	parser.set_defaults(dev_always=False)
	parser.add_argument('-test_always', dest='test_always', action='store_true', help='Evaluate on Test set at each epoch')
	parser.add_argument('-no-test_always', dest='test_always', action='store_false', help='Evaluate on Test set in the end')
	parser.set_defaults(test_always=False)
	parser.add_argument('-model_selector_set', type=str, default='gen', choices=['val', 'test', 'gen'], help='Dataset whose evaluation performance to use for model selection')
	parser.add_argument('-eval_last_n', type=int, default= 1, help='Last n epochs to be evaluated')

	# Run Config
	parser.add_argument('-pretrained_model_name', type=str, default='none', help='name of pretrained model to load. Keep none if no pretrained model')
	parser.add_argument('-finetune_data_voc', type=str, default='none', help='finetuning dataset whose voc needs to be considered now. Keep none if voc not to be added')

	parser.add_argument('-run_name', type=str, default='debug', help='run name for logs')
	parser.add_argument('-project_name', type=str, default='scan-trial', help='Name of the project')
	parser.add_argument('-dataset', type=str, default='cogs', help='Dataset')

	parser.add_argument('-display_freq', type=int, default= 10000, help='number of batches after which to display samples')
	parser.add_argument('-outputs', dest='outputs', action='store_true', help='Show full validation outputs')
	parser.add_argument('-no-outputs', dest='outputs', action='store_false', help='Do not show full validation outputs')
	parser.set_defaults(outputs=True)
	parser.add_argument('-results', dest='results', action='store_true', help='Store results')
	parser.add_argument('-no-results', dest='results', action='store_false', help='Do not store results')
	parser.set_defaults(results=True)

	# Meta Attributes
	parser.add_argument('-vocab_size', type=int, default=30000, help='Vocabulary size to consider')

	# Device Configuration
	parser.add_argument('-gpu', type=int, default=2, help='Specify the gpu to use')
	parser.add_argument('-early_stopping', type=int, default=200, help='Early Stopping after n epoch')
	parser.add_argument('-seed', type=int, default=6174, help='Default seed to set')
	parser.add_argument('-logging', type=int, default=1, help='Set to 0 if you do not require logging')
	parser.add_argument('-ckpt', type=str, default='model', help='Checkpoint file name')
	parser.add_argument('-save_model', dest='save_model',action='store_true', help='To save the model')
	parser.add_argument('-no-save_model', dest='save_model', action='store_false', help='Dont save the model')
	parser.set_defaults(save_model=False)

	# LSTM parameters
	parser.add_argument('-emb2_size', type=int, default=512, help='Embedding dimensions of outputs')
	parser.add_argument('-cell_type', type=str, default='lstm', help='RNN cell for encoder and decoder, default: lstm')

	parser.add_argument('-use_attn', dest='use_attn',action='store_true', help='To use attention mechanism?')
	parser.add_argument('-no-attn', dest='use_attn', action='store_false', help='Not to use attention mechanism?')
	parser.set_defaults(use_attn=True)

	parser.add_argument('-attn_type', type=str, default='general', help='Attention mechanism: (general, concat), default: general')
	parser.add_argument('-hidden_size', type=int, default=512, help='Number of hidden units in each layer')
	parser.add_argument('-depth', type=int, default=2, help='Number of layers in each encoder and decoder')
	parser.add_argument('-dropout', type=float, default=0.1, help= 'Dropout probability for input/output/state units (0.0: no dropout)')
	parser.add_argument('-max_length', type=int, default=60, help='Specify max decode steps: Max length string to output')
	parser.add_argument('-init_range', type=float, default=0.08, help='Initialization range for seq2seq model')
	parser.add_argument('-bidirectional', dest='bidirectional', action='store_true', help='Bidirectionality in LSTMs')
	parser.add_argument('-no-bidirectional', dest='bidirectional', action='store_false', help='Bidirectionality in LSTMs')
	parser.set_defaults(bidirectional=True)
	parser.add_argument('-lr', type=float, default=2, help='Learning rate')
	parser.add_argument('-warmup', type=float, default=0.1, help='Proportion of training to perform linear learning rate warmup for')
	parser.add_argument('-max_grad_norm', type=float, default=5.0, help='Clip gradients to this norm')
	parser.add_argument('-batch_size', type=int, default=128, help='Batch size')
	parser.add_argument('-epochs', type=int, default=50, help='Maximum # of training epochs')
	parser.add_argument('-opt', type=str, default='adam', choices=['adam', 'adadelta', 'sgd', 'asgd'], help='Optimizer for training')
	parser.add_argument('-teacher_forcing_ratio', type=float, default=0.9, help='Teacher forcing ratio')

	# Embeddings
	parser.add_argument('-embedding', type=str, default='random', choices=['bert', 'roberta', 'word2vec', 'random'], help='Embeddings')
	parser.add_argument('-word2vec_bin', type=str, default='/datadrive/global_files/GoogleNews-vectors-negative300.bin', help='Binary file of word2vec')
	parser.add_argument('-emb1_size', type=int, default=512, help='Embedding dimensions of inputs')
	parser.add_argument('-emb_name', type=str, default='roberta-base', choices=['bert-base-uncased', 'roberta-base'], help='Which pre-trained model')
	parser.add_argument('-emb_lr', type=float, default=2, help='Larning rate to train embeddings')
	parser.add_argument('-freeze_emb', dest='freeze_emb', action='store_true', help='Freeze embedding weights')
	parser.add_argument('-no-freeze_emb', dest='freeze_emb', action='store_false', help='Train embedding weights')
	parser.set_defaults(freeze_emb=False)
	parser.add_argument('-freeze_emb2', dest='freeze_emb2', action='store_true', help='Freeze output embedding weights')
	parser.add_argument('-no-freeze_emb2', dest='freeze_emb2', action='store_false', help='Train output embedding weights')
	parser.set_defaults(freeze_emb2=False)
	parser.add_argument('-freeze_lstm_encoder', dest='freeze_lstm_encoder', action='store_true', help='Freeze lstm Encoder weights')
	parser.add_argument('-no-freeze_lstm_encoder', dest='freeze_lstm_encoder', action='store_false', help='Train lstm Encoder weights')
	parser.set_defaults(freeze_lstm_encoder=False)
	parser.add_argument('-freeze_lstm_decoder', dest='freeze_lstm_decoder', action='store_true', help='Freeze lstm Decoder weights')
	parser.add_argument('-no-freeze_lstm_decoder', dest='freeze_lstm_decoder', action='store_false', help='Train lstm Decoder weights')
	parser.set_defaults(freeze_lstm_decoder=False)
	parser.add_argument('-freeze_fc', dest='freeze_fc', action='store_true', help='Freeze FC weights')
	parser.add_argument('-no-freeze_fc', dest='freeze_fc', action='store_false', help='Train FC weights')
	parser.set_defaults(freeze_fc=False)

	parser.add_argument('-show_train_acc', dest='show_train_acc', action='store_true', help='Calculate the train accuracy')
	parser.add_argument('-no-show_train_acc', dest='show_train_acc', action='store_false', help='Don\'t calculate the train accuracy')
	parser.set_defaults(show_train_acc=False)

	#Conf parameters
	parser.add_argument('-conf', type = str, default = 'posterior', choices = ["posterior", "similarity"], help = 'Confidence estimation criteria to use, ["posterior", "similarity"]')
	parser.add_argument('-sim_criteria', type = str, default = 'bleu', choices = ['bert_score', 'bleu_score'], help = 'Only applicable if similarity based criteria is selected for confidence.')
	parser.add_argument('-adv', action = 'store_true', help = 'If dealing with out of distribution examples')
	
	return parser
