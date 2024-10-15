from pymonntorch import Behavior

class TimeResolution(Behavior):
	def initialize(self, network):
		network.dt = self.parameter("dt", 1)
