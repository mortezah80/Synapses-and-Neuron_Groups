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
		k = self.parameter("v_init", default="normal(0.3, 0.05)")

		# ng.threshold = ng.vector(mode='init_threshold')
		ng.v = ng.vector(mode=k)
		ng.spike = ng.v >= self.threshold
		ng.v[ng.spike] = self.u_back


	def forward(self, ng):

		currents = self.R * ng.I
		# dynamic
		leakage = -(ng.v - self.u_rest)
		ng.v += ((leakage + currents) / self.tau) * ng.network.dt
		# firing
		ng.spike = ng.v >= self.threshold
		#reset
		ng.v[ng.spike] = self.u_back
