from pymonntorch import *
from lif3 import LIF
from dt import TimeResolution
from current import SetCurrent
from syn import SynFun, InpSyn
import torch

torch.manual_seed(73)

# settings = {"def_type": torch.float32, "device": 'cpu'}
net = Network(behavior={1: TimeResolution(dt=1)})

pop1 = NeuronGroup(size=10,
                   net=net,
                   behavior={2: SetCurrent(value=7),
                             5: LIF(R=5,
                                    threshold=-37,
                                    u_rest=-67,
                                    u_back=-75,
                                    tau=10),
                             7: Recorder(tag="pop1_rec", variables=["v"]),
                             8: EventRecorder(tag="pop1_event", variables=["spike"])})

pop2 = NeuronGroup(size=10,
                   net=net,
                   behavior={2: SetCurrent(value=5),
                             4: InpSyn(),
                             5: LIF(R=5,
                                    threshold=-37,
                                    u_rest=-67,
                                    u_back=-75,
                                    tau=10),
                             7: Recorder(tag="pop2_rec", variables=["v"]),
                             8: EventRecorder(tag="pop2_event", variables=["spike"])})

syn = SynapseGroup(tag="normal", net=net, src=pop1, dst=pop2, behavior={3: SynFun(j0=5)})

net.initialize()
net.simulate_iterations(100)

import matplotlib.pyplot as plt

plt.plot(net["pop1_rec", 0].variables["v"][:,:3])
plt.show()

plt.plot(net["pop2_rec", 0].variables["v"][:,:3])
plt.show()

plt.scatter(net["pop1_event",0]["spike",0][:,0],net["pop1_event",0]["spike",0][:,1])
plt.scatter(net["pop2_event",0]["spike",0][:,0],net["pop2_event",0]["spike",0][:,1]+10)
plt.show()
