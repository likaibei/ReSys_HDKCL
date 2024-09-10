import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, Denoise, GaussianDiffusion
from DataHandler import DataHandler
import numpy as np
import pickle
from Utils.Utils import *
import os
import random
import torch.nn as nn
from geoopt.optim import RiemannianAdam

class Coach:
	def __init__(self, handler, train_data, test_data):
		self.handler = handler
		self.train_data =train_data
		self.test_data = test_data
		self.decay = args.l2

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()
	
	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret
	
	def run(self):
		self.prepareModel()
		log('Model Prepared')
		log('Model Initialized')

		recallMax = 0
		ndcgMax = 0
		bestEpoch = 0

		for ep in range(0, args.epoch):
			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, tstFlag))
			if tstFlag:
				reses = self.testEpoch()
				if (reses['Recall'] > recallMax):
					recallMax = reses['Recall']
					ndcgMax = reses['NDCG']
					bestEpoch = ep
				log(self.makePrint('Test', ep, reses, tstFlag))
			print()
		print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , NDCG : ', ndcgMax)

	def prepareModel(self):
		self.model = Model(self.handler).cuda()
		# self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
		self.opt = RiemannianAdam(self.model.parameters(), lr=args.lr, weight_decay=0)

		self.diffusion_model = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps).cuda()
		out_dims = eval(args.dims) + [args.entity_n]
		in_dims = out_dims[::-1]
		self.denoise_model = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm, mask_BCE = args.mask_BCE).cuda()
		self.denoise_opt = torch.optim.Adam(self.denoise_model.parameters(), lr=args.lr, weight_decay=0)

	def trainEpoch(self):
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		epLoss, epRecLoss, epClLoss = 0, 0, 0
		epDiLoss, epUKLoss = 0, 0
		steps = trnLoader.dataset.__len__() // args.batch

		diffusionLoader = self.handler.diffusionLoader

		for i, batch in enumerate(diffusionLoader):
			batch_item, batch_index, batch_idx = batch
			lpos, dpos = batch_index
			batchMask = np.ones_like(batch_item)

			if args.mask_BCE:
				for itemBatch in range(len(batchMask)):
					lenLL = lpos[itemBatch]
					LL = dpos[itemBatch]
					# TODO extend to mask more items
					if np.random.randint(1, 10) % 3 == 0:
						mPos1 = LL[random.randint(0, lenLL)]
						batchMask[itemBatch][int(mPos1.item())] = 0


			maskedItem = np.ones_like(batchMask) - batchMask
			maskedBatch = torch.from_numpy(maskedItem) * batch_item
			remaindItem = torch.from_numpy(batchMask) * batch_item
			maskedBatch = maskedBatch.cuda()
			remaindItem = remaindItem.cuda()
			batch_item = batch_item.cuda()

			ui_matrix = self.handler.ui_matrix
			iEmbeds = self.model.getEntityEmbeds().detach()
			uEmbeds = self.model.getUserEmbeds().detach()

			self.denoise_opt.zero_grad()

			diff_loss, ukgc_loss = self.diffusion_model.training_losses(self.denoise_model, remaindItem, ui_matrix, uEmbeds, iEmbeds, batch_idx, maskedBatch)
			angle_loss = self.model.angle_loss(iEmbeds,uEmbeds)
			# loss = diff_loss.mean() * (1-args.e_loss) + ukgc_loss.mean() * args.e_loss + angle_loss
			loss = diff_loss.mean() * (1 - args.e_loss) + angle_loss.mean() * args.e_loss

			epDiLoss += diff_loss.mean().item()
			epUKLoss += ukgc_loss.mean().item()

			loss.backward()

			self.denoise_opt.step()

			log('Diffusion Step %d/%d' % (i, diffusionLoader.dataset.__len__() // args.batch), save=False, oneline=True)

		log('')
		log('Start to re-build kg')

		e_idxlist = list(range(train_data.shape[0]))

		with torch.no_grad():
			denoised_edges = []
			h_list = []
			t_list = []

			for batch_in, batch in enumerate(diffusionLoader):
				batch_item, batch_index,batch_idx = batch
				# batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

				batchMask = np.ones_like(batch_item)
				maskedItem = np.ones_like(batchMask) - batchMask
				maskedBatch = torch.from_numpy(maskedItem) * batch_item
				remaindItem = torch.from_numpy(batchMask) * batch_item
				train = train_data.tocsr()
				his_data = train[e_idxlist[batch_in * args.batch:batch_in * args.batch + len(batch)]]
				maskedBatch = maskedBatch.cuda()
				remaindItem = remaindItem.cuda()
				batch_item =batch_item.cuda()

				denoised_batch = self.diffusion_model.p_sample(self.denoise_model, batch_item, args.sampling_steps, maskedBatch)
				top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)
				for i in range(batch_idx.shape[0]):
					for j in range(indices_[i].shape[0]):
						h_list.append(batch_idx[i])
						t_list.append(indices_[i][j])

			edge_set = set()
			for index in range(len(h_list)):
				edge_set.add((int(h_list[index].cpu().numpy()), int(t_list[index].cpu().numpy())))
			for index in range(len(h_list)):
				if (int(t_list[index].cpu().numpy()), int(h_list[index].cpu().numpy())) not in edge_set:
					h_list.append(t_list[index])
					t_list.append(h_list[index])

			relation_dict = self.handler.relation_dict
			for index in range(len(h_list)):
				try:
					denoised_edges.append([h_list[index], t_list[index], relation_dict[int(h_list[index].cpu().numpy())][int(t_list[index].cpu().numpy())]])
				except Exception:
					continue
			graph_tensor = torch.tensor(denoised_edges)
			index_ = graph_tensor[:, :-1]
			type_ = graph_tensor[:, -1]
			denoisedKG = (index_.t().long().cuda(), type_.long().cuda())
		
		log('KG built!')

		with torch.no_grad():
			index_, type_ = denoisedKG
			mask = ((torch.rand(type_.shape[0]) + args.keepRate).floor()).type(torch.bool)
			denoisedKG = (index_[:, mask], type_[mask])
			self.generatedKG = denoisedKG

		for i, tem in enumerate(trnLoader):
			ancs, poss, negs = tem
			ancs = ancs.long().cuda()
			poss = poss.long().cuda()
			negs = negs.long().cuda()

			self.opt.zero_grad()

			if args.cl_pattern == 0:
				usrEmbeds, itmEmbeds = self.model(self.handler.torchBiAdj, denoisedKG)
			else:
				usrEmbeds, itmEmbeds = self.model(self.handler.torchBiAdj)
			ancEmbeds = usrEmbeds[ancs]
			posEmbeds = itmEmbeds[poss]
			negEmbeds = itmEmbeds[negs]

			# scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
			# bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
			bprLoss = self.create_inner_bpr_loss(ancEmbeds, posEmbeds, negEmbeds)
			regLoss = calcRegLoss(self.model) * args.reg

			if args.cl_pattern == 0:
				usrEmbeds_kg, itmEmbeds_kg = self.model(self.handler.torchBiAdj)
			else:
				usrEmbeds_kg, itmEmbeds_kg = self.model(self.handler.torchBiAdj, denoisedKG)
			denoisedKGEmbeds = torch.concat([usrEmbeds, itmEmbeds], axis=0)
			kgEmbeds = torch.concat([usrEmbeds_kg, itmEmbeds_kg], axis=0)

			clLoss = (contrastLoss(kgEmbeds[args.user:], denoisedKGEmbeds[args.user:], poss, args.temp) + contrastLoss(kgEmbeds[:args.user], denoisedKGEmbeds[:args.user], ancs, args.temp)) * args.ssl_reg

			loss = bprLoss + regLoss + clLoss

			epLoss += loss.item()
			epRecLoss += bprLoss.item()
			epClLoss += clLoss.item()

			loss.backward()
			self.opt.step()

			log('Step %d/%d: loss = %.3f, regLoss = %.3f' % (i, steps, loss, regLoss), save=False, oneline=True)

		ret = dict()
		ret['Loss'] = epLoss / steps
		ret['recLoss'] = epRecLoss / steps
		ret['clLoss'] = epClLoss / steps
		ret['diLoss'] = epDiLoss / diffusionLoader.dataset.__len__()
		ret['UKGCLoss'] = epUKLoss / diffusionLoader.dataset.__len__()
		return ret

	def testEpoch(self):
		tstLoader = self.handler.tstLoader
		epRecall, epNdcg = [0] * 2
		i = 0
		num = tstLoader.dataset.__len__()
		steps = num // args.tstBat

		with torch.no_grad():
			if args.cl_pattern == 0:
				denoisedKG = self.generatedKG
				usrEmbeds, itmEmbeds = self.model(self.handler.torchBiAdj, mess_dropout=False, kg=denoisedKG)
			else:
				usrEmbeds, itmEmbeds = self.model(self.handler.torchBiAdj, mess_dropout=False)

		for usr, trnMask in tstLoader:
			i += 1
			usr = usr.long().cuda()
			trnMask = trnMask.cuda()

			allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
			_, topLocs = t.topk(allPreds, args.topk)
			recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
			epRecall += recall
			epNdcg += ndcg
			log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		return ret

	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = 0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
			recall = dcg = 0
			for val in temTstLocs:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			allRecall += recall
			allNdcg += ndcg
		return allRecall, allNdcg

	def create_inner_bpr_loss(self, users, pos_items, neg_items):
		batch_size = users.shape[0]
		pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
		neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

		cf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
		# cul regularizer
		regularizer = (torch.norm(users) ** 2
					   + torch.norm(pos_items) ** 2
					   + torch.norm(neg_items) ** 2) / 2
		emb_loss = self.decay * regularizer / batch_size

		return cf_loss + emb_loss

	
def seed_it(seed):
	random.seed(seed)
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True 
	torch.backends.cudnn.enabled = True
	torch.manual_seed(seed)

if __name__ == '__main__':
	seed_it(args.seed)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	logger.saveDefault = True
	
	log('Start')
	handler = DataHandler()
	train_data, test_data = handler.LoadData()
	log('Load Data')

	coach = Coach(handler,train_data, test_data)
	coach.run()

