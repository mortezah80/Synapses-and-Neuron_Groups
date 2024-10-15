from pymonntorch import Behavior
import torch
import random


class SynFun(Behavior):
	def initialize(self, sg):
		sg.j0 = self.parameter("j0")
		sg.W = sg.matrix(mode="uniform")
		# for i in range(min(len(sg.W[0]),len(sg.W))):
		# 	sg.W[i][i] = 0
		sg.I = sg.dst.vector()

	def forward(self, sg):
		sg.I = torch.sum(sg.W[sg.src.spike], axis=0)*5

class full_fix_SynFun(Behavior):
	def initialize(self, sg):
		sg.j0 = self.parameter("j0")
		sg.W = sg.matrix(mode=sg.j0/sg.src.size)
		# sg.W = sg.matrix(mode="uniform")
		# for i in range(min(len(sg.W[0]),len(sg.W))):
		# 	sg.W[i][i] = 0
		sg.I = sg.dst.vector()

	def forward(self, sg):
		sg.I = torch.sum(sg.W[sg.src.spike], axis=0)


class full_guassian_SynFun(Behavior):
	def initialize(self, sg):
		sg.j0 = self.parameter("j0")
		sg.sigma = self.parameter("sigma")

		mean = sg.j0/sg.src.size
		variance = sg.sigma / sg.src.size
		sg.W = sg.matrix(mode=f"normal({mean}, {variance})")
		# for i in range(min(len(sg.W[0]),len(sg.W))):
		# 	sg.W[i][i] = 0
		sg.I = sg.dst.vector()

	def forward(self, sg):
		sg.I = torch.sum(sg.W[sg.src.spike], axis=0)



class random_guassian_SynFun(Behavior):
	def initialize(self, sg):
		sg.j0 = self.parameter("j0")
		sg.p = self.parameter("p")

		sg.W = sg.matrix(mode=0)
		print(sg.W)
		for row in range(len(sg.W)):
			for column in range(len(sg.W[row])):
				rand = random.random()
				if rand < sg.p:
					sg.W[row][column] = sg.j0 / (sg.p * sg.src.size)

		print(sg.W)
		# for i in range(min(len(sg.W[0]),len(sg.W))):
		# 	sg.W[i][i] = 0
		sg.I = sg.dst.vector()

	def forward(self, sg):
		sg.I = torch.sum(sg.W[sg.src.spike], axis=0)


class Init_synapse(Behavior):
	def initialize(self, synapse):
		synapse.j0 = self.parameter("j0")
		synapse.coef = self.parameter("coef")
		synapse.sigma = self.parameter("sigma")
		synapse.p = self.parameter("p")



class InpSyn(Behavior):

	def initialize(self, ng):

		for synapse in ng.afferent_synapses['GLUTAMATE']:
			synapse.W = synapse.matrix(mode='uniform', density=0.3)
			synapse.W = synapse.W / synapse.src.size
			synapse.I = synapse.dst.vector()

		for synapse in ng.afferent_synapses['GABA']:
			synapse.W = - synapse.matrix(mode='uniform', density=0.3)
			synapse.W = synapse.W / synapse.src.size
			synapse.I = synapse.dst.vector()

		for synapse in ng.afferent_synapses['GLUTAMATE_Inner']:
			synapse.W = synapse.matrix(mode='uniform', density=0.3)
			synapse.W = synapse.W / synapse.src.size
			print(synapse.W.shape, synapse.W)
			for i in range(min(len(synapse.W[0]), len(synapse.W))):
				synapse.W[i][i] = 0
			synapse.I = synapse.dst.vector()

		for synapse in ng.afferent_synapses['GABA_Inner']:
			synapse.W = - synapse.matrix(mode='uniform', density=0.3)
			synapse.W = synapse.W / synapse.src.size
			print(synapse.W.shape, synapse.W)
			for i in range(min(len(synapse.W[0]), len(synapse.W))):
				synapse.W[i][i] = 0
			synapse.I = synapse.dst.vector()

	def synapse_current(self, synapse):
		synapse.I = torch.sum(synapse.W[synapse.src.spike], axis=0) * synapse.coef

	def forward(self, ng):

		for syn in ng.afferent_synapses["All"]:
			self.synapse_current(syn)
			ng.I += syn.I
			# ng.I += ng.vector(mode="normal(0,5)")

class InpSyn_full_fix(InpSyn):

	def initialize(self, ng):
		for synapse in ng.afferent_synapses['GLUTAMATE']:
			synapse.W = synapse.matrix(mode=synapse.j0/synapse.src.size)
			synapse.I = synapse.dst.vector()

		for synapse in ng.afferent_synapses['GABA']:
			synapse.W = - synapse.matrix(mode=synapse.j0/synapse.src.size)
			synapse.I = synapse.dst.vector()

		for synapse in ng.afferent_synapses['GLUTAMATE_Inner']:
			synapse.W = synapse.matrix(mode=synapse.j0/synapse.src.size)
			print(synapse.W.shape, synapse.W)
			for i in range(min(len(synapse.W[0]), len(synapse.W))):
				synapse.W[i][i] = 0
			synapse.I = synapse.dst.vector()

		# for synapse in ng.afferent_synapses['GABA_Inner']:
		# 	synapse.W = - synapse.matrix(mode=synapse.j0/synapse.src.size)
		# 	print(synapse.W.shape, synapse.W)
		# 	for i in range(min(len(synapse.W[0]), len(synapse.W))):
		# 		synapse.W[i][i] = 0
		# 	synapse.I = synapse.dst.vector()


class InpSyn_full_gussian(InpSyn):

	def initialize(self, ng):
		for synapse in ng.afferent_synapses['GLUTAMATE']:
			mean = synapse.j0 / synapse.src.size
			variance = synapse.sigma / synapse.src.size
			synapse.W = synapse.matrix(mode=f"normal({mean}, {variance})")
			synapse.I = synapse.dst.vector()

		for synapse in ng.afferent_synapses['GABA']:
			mean = synapse.j0 / synapse.src.size
			variance = synapse.sigma / synapse.src.size
			synapse.W = - synapse.matrix(mode=f"normal({mean}, {variance})")
			synapse.I = synapse.dst.vector()

		for synapse in ng.afferent_synapses['GLUTAMATE_Inner']:
			mean = synapse.j0 / synapse.src.size
			variance = synapse.sigma / synapse.src.size
			synapse.W = synapse.matrix(mode=f"normal({mean}, {variance})")
			print(synapse.W.shape, synapse.W)
			for i in range(min(len(synapse.W[0]), len(synapse.W))):
				synapse.W[i][i] = 0
			synapse.I = synapse.dst.vector()

		# for synapse in ng.afferent_synapses['GABA_Inner']:
		# 	mean = synapse.j0 / synapse.src.size
		# 	variance = synapse.sigma / synapse.src.size
		# 	synapse.W = - synapse.matrix(mode=f"normal({mean}, {variance})")
		# 	print(synapse.W.shape, synapse.W)
		# 	for i in range(min(len(synapse.W[0]), len(synapse.W))):
		# 		synapse.W[i][i] = 0
		# 	synapse.I = synapse.dst.vector()


class InpSyn_random_gussian(InpSyn):

	def initialize(self, ng):
		for synapse in ng.afferent_synapses['GLUTAMATE']:
			synapse.W = synapse.matrix(mode=0)
			for row in range(len(synapse.W)):
				for column in range(len(synapse.W[row])):
					rand = random.random()
					if rand < synapse.p:
						synapse.W[row][column] = synapse.j0 / (synapse.p * synapse.src.size)

			print(synapse.W)
			synapse.I = synapse.dst.vector()

		for synapse in ng.afferent_synapses['GABA']:
			synapse.W = synapse.matrix(mode=0)
			for row in range(len(synapse.W)):
				for column in range(len(synapse.W[row])):
					rand = random.random()
					if rand < synapse.p:
						synapse.W[row][column] = synapse.j0 / (synapse.p * synapse.src.size)
			synapse.W = - synapse.W
			print(synapse.W)
			synapse.I = synapse.dst.vector()

		for synapse in ng.afferent_synapses['GLUTAMATE_Inner']:
			synapse.W = synapse.matrix(mode=0)
			for row in range(len(synapse.W)):
				for column in range(len(synapse.W[row])):
					rand = random.random()
					if rand < synapse.p:
						synapse.W[row][column] = synapse.j0 / (synapse.p * synapse.src.size)
			print(synapse.W)
			print(synapse.W.shape, synapse.W)
			for i in range(min(len(synapse.W[0]), len(synapse.W))):
				synapse.W[i][i] = 0
			synapse.I = synapse.dst.vector()

		# for synapse in ng.afferent_synapses['GABA_Inner']:
		# 	synapse.W = synapse.matrix(mode=0)
		# 	for row in range(len(synapse.W)):
		# 		for column in range(len(synapse.W[row])):
		# 			rand = random.random()
		# 			if rand < synapse.p:
		# 				synapse.W[row][column] = synapse.j0 / (synapse.p * synapse.src.size)
		# 	synapse.W = - synapse.W
		# 	print(synapse.W)
		# 	print(synapse.W.shape, synapse.W)
		# 	for i in range(min(len(synapse.W[0]), len(synapse.W))):
		# 		synapse.W[i][i] = 0
		# 	synapse.I = synapse.dst.vector()
