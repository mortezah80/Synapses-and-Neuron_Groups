from pymonntorch import Behavior
import numpy as np

np.random.seed(60)
class SetCurrent(Behavior):
	def initialize(self, ng):
		self.offset = self.parameter("value")
		self.s_current = self.parameter("s_current")
		self.e_current = self.parameter("e_current")
		ng.I = ng.vector(self.offset)
		ng.I_P = ng.vector(self.offset)
		self.current_value = 0
	def forward(self, ng):
		# ng.I = ng.vector(self.offset)
		if 40 < ng.network.iteration <42:
			self.current_value +=10
		# if 64 < ng.network.iteration < 66:
		# 	self.current_value += 20
		step = np.random.randn() # Generate a random step
		self.current_value = max(0, self.current_value + step)
		ng.I = ng.vector(mode=self.current_value)
		ng.I_P = ng.vector(mode=self.current_value)
		# ng.I += ng.vector(mode="normal(0,5)")


class SetCurrent_inh(Behavior):
	def initialize(self, ng):
		self.offset = self.parameter("value")
		self.s_current = self.parameter("s_current")
		self.e_current = self.parameter("e_current")
		ng.I = ng.vector(self.offset)
		ng.I_P = ng.vector(self.offset)
		self.current_value = 0
	def forward(self, ng):
		ng.I = ng.vector(self.offset)