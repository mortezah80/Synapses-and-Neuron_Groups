from pymonntorch import *


class LIF(Behavior):
    def initialize(self, ng):
        self.R = self.parameter("R", None)      
        self.tau = self.parameter("tau", None)
        self.u_rest = self.parameter("u_rest", None)
        self.u_back = self.parameter("u_back", None)
        self.threshold = self.parameter("threshold", None)
        ratio = self.parameter("scale_ratio", 1.10)

        # ng.v = ng.vector("uniform")
        # ng.v += ng.vector(mode=self.u_reset)
        ng.v = ng.vector("uniform") * (self.threshold - self.u_back) * ratio
        ng.v += self.u_back

        ng.spike = ng.v > self.threshold

    
    def forward(self, ng):
        leakage = -(ng.v - self.u_rest)
        inp_u = self.R * ng.I #TODO
        ng.v += ((leakage + inp_u) / self.tau) * ng.network.dt

        ng.spike = ng.v > self.threshold
        ng.v[ng.spike] = self.u_back
        # ng.v[ng.spike] = self.u_reset

