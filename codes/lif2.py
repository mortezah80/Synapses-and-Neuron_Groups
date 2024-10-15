import torch
import math

from pymonntorch import Behavior


class LIF(Behavior):
	def initialize(self, ng):
		self.tau = self.parameter("tau")
		self.u_rest = self.parameter("u_rest")
		self.u_reset = self.parameter("u_reset")
		self.u_back = self.parameter("u_back")
		self.threshold = self.parameter("threshold")
		self.R = self.parameter("R")
		self.refractory_T = self.parameter("refractory_T")
		self.refractor_bool = False
		self.last_spike = 0
		k = self.parameter("v_init", default="normal(0.3, 0.05)")

		# ng.threshold = ng.vector(mode='init_threshold')
		ng.v = ng.vector(mode=k)
		ng.spike = ng.v >= self.threshold
		ng.v[ng.spike] = self.u_back


	def forward(self, ng):

		if self.refractor_bool:
			currents = 0
			if ng.network.iteration * ng.network.dt - self.last_spike > self.refractory_T:
				self.refractor_bool = False
		else:
			currents = self.R * ng.I
		# dynamic
		leakage = -(ng.v - self.u_rest)

		ng.v += ((leakage + currents) / self.tau) * ng.network.dt

		# firing
		ng.spike = ng.v >= self.threshold

		#reset
		ng.v[ng.spike] = self.u_back

		if (ng.spike):
			self.refractor_bool = True
			self.last_spike = ng.network.iteration * ng.network.dt
			# W flow
			# self.spike_number +=1
class ELIF(LIF):
	def initialize(self, ng):
		self.tau = self.parameter("tau")
		self.u_rest = self.parameter("u_rest")
		self.u_reset = self.parameter("u_reset")
		self.u_back = self.parameter("u_back")
		self.threshold = self.parameter("threshold")
		self.R = self.parameter("R")
		self.refractory_T = self.parameter("refractory_T")
		self.refractor_bool = False
		self.last_spike = 0

		k = self.parameter("v_init", default="normal(0.3, 0.05)")

		# ng.threshold = ng.vector(mode='init_threshold')
		ng.v = ng.vector(mode=k)
		ng.spike = ng.v >= self.u_reset
		ng.v[ng.spike] = self.u_back

		self.delta_T = self.parameter("delta_T")

	def forward(self, ng):
		# leakage = -(ng.v - self.u_rest)
		# currents = self.R * ng.I
		# exp_term = self.delta_T * math.exp((ng.v - self.threshold) / self.delta_T)
		# ng.v += ((leakage + currents + exp_term) / self.tau) * ng.network.dt
		# # firing
		# ng.spike = ng.v >= self.u_reset
		#
		#
		# # reset
		# ng.v[ng.spike] = self.u_back


		# refractory added

		if self.refractor_bool:
			ng.spike = ng.vector(mode=0)
			if ng.network.iteration * ng.network.dt - self.last_spike > self.refractory_T:
				self.refractor_bool = False

		else:
			leakage = -(ng.v - self.u_rest)
			currents = self.R * ng.I
			exp_term = 0
			if ng.v >= self.threshold :
				exp_term = self.delta_T * math.exp((ng.v - self.threshold) / self.delta_T)
			else:
				exp_term = 0
			ng.v += ((leakage + currents + exp_term) / self.tau) * ng.network.dt

			# firing
			ng.spike = ng.v >= self.u_reset

			# reset
			ng.v[ng.spike] = self.u_back
			if (ng.spike):
				self.refractor_bool = True
				self.last_spike = ng.network.iteration * ng.network.dt


class AELIF(LIF):
	def initialize(self, ng):
		ELIF.initialize(self,ng)
		self.a = self.parameter("a")
		self.b = self.parameter("b")
		self.tau_w = self.parameter("tau_w")
		ng.w = 0
		self.spike_number = 0
		ng.first_term = ng.vector(mode=0)
		ng.second_term = ng.vector(mode=0)
		ng.third_term = ng.vector(mode=0)


	def forward(self, ng):

		if self.refractor_bool:
			ng.spike = ng.vector(mode=0)
			if ng.network.iteration * ng.network.dt - self.last_spike > self.refractory_T:
				self.refractor_bool = False

		else:
			leakage = -(ng.v - self.u_rest)
			currents = self.R * ng.I
			exp_term = 0
			if ng.v >= self.threshold :
				exp_term = self.delta_T * math.exp((ng.v - self.threshold) / self.delta_T)
			else:
				exp_term = 0

			leakge_w = self.a * (ng.v - self.u_rest) - ng.w
			ng.v += ((leakage + currents + exp_term + (-(self.R * ng.w))) / self.tau) * ng.network.dt

			# firing
			ng.spike = ng.v >= self.u_reset
			# reset
			ng.v[ng.spike] = self.u_back


			adapt_term = 0
			# adapt would be zero and if spike would be b * tau_w
			if (ng.spike):
				self.refractor_bool = True
				self.last_spike = ng.network.iteration * ng.network.dt
				# W flow
				# self.spike_number +=1
				adapt_term = self.b * self.tau_w

			ng.w += ((leakge_w + adapt_term) / self.tau_w) * ng.network.dt
			ng.first_term = ng.vector(mode=(self.a * (ng.v - self.u_rest)).item())
			# ng.second_term = ng.vector(mode=a)

			ng.third_term = ng.vector(mode=adapt_term)

				# ng.v += ((-(self.R * self.w)) / self.tau) * ng.network.dt