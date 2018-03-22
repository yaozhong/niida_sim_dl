from __future__ import division

import sys
from load_data import *
from dl_model import *
from chainer.datasets import tuple_dataset

import argparse
import numpy as np
import cupy as cp

"""
Implemented functions:
[o] 1. Loading GTF file, do the option selection of the target regions. 
[o] 2. Extract the sequences of the given region. 
3. Transfer sequences to k-mers and prepare the training format. 
"""
def trainDL():

	parser = argparse.ArgumentParser(description='coding and non-coding sequence distinguish')

	## training options
	parser.add_argument('--batchsize', '-mb', type=int, default=128, help='Number of bins in each mini-batch')
	parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of iters over the dataset for train')
	parser.add_argument('--out', '-o', default='train_curve', help='Directory to output the result')
	parser.add_argument('--modelPath', '-mp', default='./model', help='model output path')

	parser.add_argument('--frequency', '-f', type=int, default=-1, help='Frequency of taking a snapshot')
	parser.add_argument('--resume', '-r', default='', help='Resume the raining from snapshot')

	parser.add_argument('--gpu', '-g', type=int, default=1, help='GPU ID (if none -1)')
	#parser.add_argument('--binSize', '-b', required=True, type=int, default=100, help='binSize')

	parser.add_argument('--unit', '-u', type=int, default=64, help='Number of units')
	parser.add_argument('--model', '-m', required=True, default=MLP, help='deep learning model')
	
	args = parser.parse_args()
	n_kernel = 1

	dataPath="/data/Bioinfo/niida_sim_dl/data/simlationData"
	train, test = genDLdata(dataPath, loadFromCache=True)
	## now training deep learning models from here

	print '-----------Data Loading Done---------------'
	print('\n# [Model]:{}'.format(args.model))
	print('# [GPU]: {}'.format(args.gpu))
	print('# [unit]: {}'.format(args.unit))
	print('# [Minibatch-size]: {}'.format(args.batchsize))
	print('# [epoch]: {}'.format(args.epoch))
	print('')

	model = Augmentor(globals()[args.model](500, n_kernel))

	if args.gpu >= 0:
		chainer.cuda.get_device_from_id(args.gpu).use()
		model.to_gpu()

	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)

	train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
	test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

	# setup a trainer
	updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
	trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
	trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

	trainer.extend(extensions.LogReport(log_name="train.log"))
	trainer.extend(extensions.dump_graph('main/ACC'))

	if extensions.PlotReport.available():
		trainer.extend(extensions.PlotReport(['main/ACC', 'validation/ACC'], 'epoch', file_name= "acc_curve.png"))

	trainer.extend(extensions.PrintReport(['epoch', 'main/ACC', 'main/LOSS', 'validation/main/ACC', 'validation/main/LOSS','elapsed_time']))
	trainer.extend(extensions.ProgressBar())

	## running the training process
	trainer.run()
	
	## saving the model 
	if not os.path.exists(args.modelPath):
		os.makedirs(args.modelPath)
	serializers.save_npz(args.modelPath +'/'+ outFileName +'.model', model)


if __name__ == "__main__":
	trainDL()



