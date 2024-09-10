import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
from torch_scatter import scatter_sum, scatter_softmax
import math
from Utils.hyperbolic import *
from torch_scatter import scatter_mean
init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class Model(nn.Module):
	def __init__(self, handler):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
		self.eEmbeds = nn.Parameter(init(torch.empty(args.entity_n, args.latdim)))
		self.rEmbeds = nn.Parameter(init(torch.empty(args.relation_num, args.latdim)))

		self.dropout_p = 0.5
		self.anlge_emb_dropout = nn.Dropout(p=self.dropout_p)

		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])
		self.rgat = RGAT(args.latdim, args.layer_num_kg, args.mess_dropout_rate)
		self.n_items = args.item
		self.angle_loss_w = args.angle_loss_w

		self.kg_dict = handler.kg_dict
		self.edge_index, self.edge_type = self.sampleEdgeFromDict(self.kg_dict, triplet_num=args.triplet_num)
		self.triplet_item_att = self._triplet_sampling(self.edge_index, self.edge_type).t()
	def getEntityEmbeds(self):
		return self.eEmbeds

	def getUserEmbeds(self):
		return self.uEmbeds
				
	def forward(self, adj, mess_dropout=True, kg=None):
		if kg == None:
			hids_KG = self.rgat.forward(self.eEmbeds, self.rEmbeds, [self.edge_index, self.edge_type], mess_dropout)
		else:
			hids_KG = self.rgat.forward(self.eEmbeds, self.rEmbeds, kg, mess_dropout)
						
		embeds = torch.concat([self.uEmbeds, hids_KG[:args.item, :]], axis=0)
		embedsLst = [embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		embeds = sum(embedsLst)

		return embeds[:args.user], embeds[args.user:]
	
	def sampleEdgeFromDict(self, kg_dict, triplet_num=None):
		sampleEdges = []
		for h in kg_dict:
			t_list = kg_dict[h]
			if triplet_num != -1 and len(t_list) > triplet_num:
				sample_edges_i = random.sample(t_list, triplet_num)
			else:
				sample_edges_i = t_list
			for r, t in sample_edges_i:
				sampleEdges.append([h, t, r])
		return self.getEdges(sampleEdges)

	def _triplet_sampling(self, edge_index, edge_type, rate=0.5):
		# edge_index: [2, -1]
		# edge_type: [-1]
		edge_index_t = edge_index.t()
		sample = []
		for idx, h_t in enumerate(edge_index_t):
			if (h_t[0] >= self.n_items and h_t[1] < self.n_items) or (h_t[0] < self.n_items and h_t[1] >= self.n_items):
				sample.append(idx)
		sample = torch.LongTensor(sample)

		return edge_index_t[sample]
	
	def getEdges(self, kg_edges):
		graph_tensor = torch.tensor(kg_edges)
		index = graph_tensor[:, :-1]
		type = graph_tensor[:, -1]
		return index.t().long().cuda(), type.long().cuda()

	def half_aperture(self, u):
		eps = 1e-6
		K = 0.1
		sqnu = u.pow(2).sum(dim=-1)
		sqnu.clamp_(min=0, max=1 - eps)
		return torch.asin((K * (1 - sqnu) / torch.sqrt(sqnu)).clamp(min=-1 + eps, max=1 - eps))

	def angle_at_u(self, u, v):
		eps = 1e-6
		norm_u = u.norm(2, dim=-1)
		norm_v = v.norm(2, dim=-1)
		dot_prod = (u * v).sum(dim=-1)
		edist = (u - v).norm(2, dim=-1)  # euclidean distance
		num = (dot_prod * (1 + norm_u ** 2) - norm_u ** 2 * (1 + norm_v ** 2))
		denom = (norm_u * edist * ((1 + norm_v ** 2 * norm_u ** 2 - 2 * dot_prod).clamp(min=eps).sqrt())) + eps
		return (num / denom).clamp_(min=-1 + eps, max=1 - eps).acos()

	def angle_loss(self, entity_emb, user):
		hier_hs = entity_emb[self.triplet_item_att[0]]
		hier_ts = entity_emb[self.triplet_item_att[1]]
		emb_drop = self.anlge_emb_dropout(torch.ones(size=hier_hs.shape)) * self.dropout_p  # need to tune
		emb_drop = emb_drop.cuda()
		hier_hs = hier_hs * emb_drop
		hier_ts = hier_ts * emb_drop

		loss3 = 0
		batch_size = user.shape[0]
		num = self.triplet_item_att.shape[1]
		num_x = math.ceil(num / batch_size)
		for i in range(num_x):
			hier_h = hier_hs[i * batch_size:(i + 1) * batch_size]
			hier_t = hier_ts[i * batch_size:(i + 1) * batch_size]
			angle_half = self.angle_at_u(hier_h, hier_t) - self.half_aperture(hier_h)
			angle_half[angle_half < 0] = 0
			loss3 += torch.sum(angle_half)

		loss3 = self.angle_loss_w * loss3 / num

		return loss3
	
class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return torch.spmm(adj, embeds)
		
class RGAT(nn.Module):
	def __init__(self, latdim, n_hops, mess_dropout_rate=0.4):
		super(RGAT, self).__init__()
		self.mess_dropout_rate = mess_dropout_rate
		self.W = nn.Parameter(init(torch.empty(size=(2*latdim, latdim)), gain=nn.init.calculate_gain('relu')))

		self.relu = nn.ReLU(0.2)
		self.n_hops = n_hops
		self.dropout = nn.Dropout(p=mess_dropout_rate)

	# def agg(self, entity_emb, relation_emb, kg):
	# 	edge_index, edge_type = kg
	# 	head, tail = edge_index
	# 	a_input = torch.cat([entity_emb[head], entity_emb[tail]], dim=-1)
	# 	e_input = torch.multiply(torch.mm(a_input, self.W), relation_emb[edge_type]).sum(-1)
	# 	e = self.leakyrelu(e_input)
	# 	e = scatter_softmax(e, head, dim=0, dim_size=entity_emb.shape[0])
	# 	agg_emb = entity_emb[tail] * e.view(-1, 1)
	# 	agg_emb = scatter_sum(agg_emb, head, dim=0, dim_size=entity_emb.shape[0])
	# 	agg_emb = agg_emb + entity_emb
	# 	return agg_emb
	def agg(self, entity_look_up_emb,
				   relation_lookup_emb,kg):
		edge_index, edge_type = kg
		head, tail = edge_index
		# exclude interact in IDs, remap [1, n_relations) to [0, n_relations-1)
		relation_type = edge_type - 1
		# [n_entities_, emb_dim]
		n_entities = entity_look_up_emb.shape[0]
		n_tripets = len(head)

		head_emb = entity_look_up_emb[head]
		tail_emb = entity_look_up_emb[tail]
		relation_emb = relation_lookup_emb[relation_type]

		ricci = tail_emb + relation_emb
		ricci_normalized = F.normalize(ricci.view(ricci.size(0), -1), dim=1).view_as(ricci)

		# Mobius addition

		hyper_head_emb = expmap0(head_emb)

		# map to local hyperbolic point
		hyper_tail_emb = expmap(tail_emb, hyper_head_emb)
		hyper_relation_emb = expmap(relation_emb, hyper_head_emb)
		res = project(mobius_add(hyper_tail_emb, hyper_relation_emb))
		# map to local tangent point
		res = logmap(res, hyper_head_emb) + ricci_normalized*1e-7
		res = self.relu(res)

		entity_agg = scatter_mean(src=res, index=head, dim_size=n_entities, dim=0)

		return entity_agg

		
	def forward(self, entity_emb, relation_emb, kg, mess_dropout=True):
		entity_res_emb = entity_emb
		for _ in range(self.n_hops):
			entity_emb = self.agg(entity_emb, relation_emb, kg)
			if mess_dropout:
				entity_emb = self.dropout(entity_emb)
			entity_emb = F.normalize(entity_emb)

			entity_res_emb = args.res_lambda * entity_res_emb + entity_emb
		return entity_res_emb
	def half_aperture(self, u):
		eps = 1e-6
		K = 0.1
		sqnu = u.pow(2).sum(dim=-1)
		sqnu.clamp_(min=0, max=1 - eps)
		return torch.asin((K * (1 - sqnu) / torch.sqrt(sqnu)).clamp(min=-1 + eps, max=1 - eps))

	def angle_at_u(self, u, v):
		eps = 1e-6
		norm_u = u.norm(2, dim=-1)
		norm_v = v.norm(2, dim=-1)
		dot_prod = (u * v).sum(dim=-1)
		edist = (u - v).norm(2, dim=-1)  # euclidean distance
		num = (dot_prod * (1 + norm_u ** 2) - norm_u ** 2 * (1 + norm_v ** 2))
		denom = (norm_u * edist * ((1 + norm_v ** 2 * norm_u ** 2 - 2 * dot_prod).clamp(min=eps).sqrt())) + eps
		return (num / denom).clamp_(min=-1 + eps, max=1 - eps).acos()

	def angle_loss(self, entity_emb, user):
		hier_hs = entity_emb[self.triplet_item_att[0]]
		hier_ts = entity_emb[self.triplet_item_att[1]]

		emb_drop = self.anlge_emb_dropout(torch.ones(size=hier_hs.shape)) * self.dropout_p  # need to tune
		emb_drop = emb_drop.to(self.device)
		hier_hs = hier_hs * emb_drop
		hier_ts = hier_ts * emb_drop

		loss_angle = 0
		batch_size = user.shape[0]
		num = self.triplet_item_att.shape[1]
		num_x = math.ceil(num / batch_size)
		for i in range(num_x):
			hier_h = hier_hs[i * batch_size:(i + 1) * batch_size]
			hier_t = hier_ts[i * batch_size:(i + 1) * batch_size]
			angle_half = self.angle_at_u(hier_h, hier_t) - self.half_aperture(hier_h)
			angle_half[angle_half < 0] = 0
			loss_angle += torch.sum(angle_half)

		loss_angle = self.angle_loss_w * loss_angle / num

		return loss_angle
	
class Denoise(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5, mask_BCE = False):
		super(Denoise, self).__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims
		self.time_emb_dim = emb_size
		self.norm = norm
		self.mask_BCE = mask_BCE

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

		in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

		out_dims_temp = self.out_dims

		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

		self.drop = nn.Dropout(dropout)
		self.init_weights()

	def init_weights(self):
		for layer in self.in_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)
		
		for layer in self.out_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)

		size = self.emb_layer.weight.size()
		std = np.sqrt(2.0 / (size[0] + size[1]))
		self.emb_layer.weight.data.normal_(0.0, std)
		self.emb_layer.bias.data.normal_(0.0, 0.001)

	def forward(self, x, timesteps, maskInfo, mess_dropout=True):
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
		temp = timesteps[:, None].float() * freqs[None]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		emb = self.emb_layer(time_emb)
		if self.norm:
			x = F.normalize(x)
		if mess_dropout:
			x = self.drop(x)
		if self.mask_BCE:
			h = torch.cat([x+maskInfo, emb], dim=-1)
		else:
			x = x.cuda()
			emb = emb.cuda()
			h = torch.cat([x, emb], dim=-1)
		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:
				h = torch.tanh(h)

		return h

class GaussianDiffusion(nn.Module):
	def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
		super(GaussianDiffusion, self).__init__()

		self.noise_scale = noise_scale
		self.noise_min = noise_min
		self.noise_max = noise_max
		self.steps = steps

		if noise_scale != 0:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
			if beta_fixed:
				self.betas[0] = 0.0001

			self.calculate_for_diffusion()

	def get_betas(self):
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max
		variance = np.linspace(start, end, self.steps, dtype=np.float64)
		alpha_bar = 1 - variance
		betas = []
		betas.append(1 - alpha_bar[0])
		for i in range(1, self.steps):
			betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
		return np.array(betas)
	
	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

	def p_sample(self, model, x_start, steps, batchMask=None):
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
			x_t = self.q_sample(x_start, t, batchMask)
		
		indices = list(range(self.steps))[::-1]

		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			model_mean, model_log_variance = self.p_mean_variance(model, x_t, t, batchMask)
			x_t = model_mean
		return x_t
			
	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
	
	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		arr = arr.cuda()
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)
	
	def p_mean_variance(self, model, x, t, batchmask):
		model_output = model(x, t, batchmask, False)

		model_variance = self.posterior_variance
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
		
		return model_mean, model_log_variance

	def training_losses(self, model, x_start, ui_matrix, userEmbeds, itmEmbeds, batch_idx, batchMask=None):
		batch_size = x_start.size(0)
		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
		noise = torch.randn_like(x_start)
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, ts, noise)
		else:
			x_t = x_start

		model_output = model(x_t, ts, batchMask)
		mse = self.mean_flat((x_start - model_output) ** 2)

		weight = self.SNR(ts - 1) - self.SNR(ts)
		weight = torch.where((ts == 0), 1.0, weight)

		diff_loss = weight * mse

		item_user_matrix = torch.spmm(ui_matrix, model_output[:, :args.item].t()).t()
		itmEmbeds_kg = torch.mm(item_user_matrix, userEmbeds)
		ukgc_loss = self.mean_flat((itmEmbeds_kg - itmEmbeds[batch_idx]) ** 2)


		return diff_loss, ukgc_loss

		
	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))
	
	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])