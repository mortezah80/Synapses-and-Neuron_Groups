from pymonntorch import Network, NeuronGroup, Recorder, EventRecorder, SynapseGroup
from lif_exc_inh import LIF
from dt import TimeResolution
from syn import InpSyn_full_fix, InpSyn_full_gussian, InpSyn_random_gussian, SynFun, InpSyn, Init_synapse, full_fix_SynFun, full_guassian_SynFun,random_guassian_SynFun
from current import SetCurrent
import torch
import numpy as np
from matplotlib import pyplot as plt



torch.manual_seed(73)

iteration_numbers = 200
ng1_size = 200
ng2_size = 50

net = Network(behavior={1: TimeResolution()}, dtype=torch.float64)

exc_group = NeuronGroup(
    size=ng1_size,
    net=net,
    behavior={
        2: SetCurrent(value=0),
        # 3: StepFunction(value=20, t0=5, t1=10, t2=20, t3=50, t4=70),
        # 3: StepFunction1(value=5, t0=20, t1=40),
        # 3: StepFunction2(value=220, t0=20, t1=40),
        4: InpSyn_full_gussian(),
        5: LIF(R=5,
               threshold=-37,
               u_rest=-67,
               u_back=-75,
               tau=10),
        7: Recorder(tag="ng1_rec", variables=["v"]),
        8: EventRecorder(tag="ng1_evrec", variables=["spike"])})


inh_group = NeuronGroup(
    size=ng2_size,
    net=net,
    behavior={
        2: SetCurrent(value=0),
        # 3: StepFunction(value=20, t0=5, t1=10, t2=20, t3=50, t4=70),
        # 3: StepFunction1(value=5, t0=20, t1=40),
        # 3: StepFunction2(value=220, t0=20, t1=40),
        4: InpSyn_full_gussian(),
        5: LIF(R=5,
               threshold=-37,
               u_rest=-67,
               u_back=-75,
               tau=10),
        7: Recorder(tag="ng2_rec", variables=["v", "I", "I_P"]),
        8: EventRecorder(tag="ng2_evrec", variables=["spike"])})

#
# SynapseGroup(net=net, src=ng1, dst=ng2, behavior={
#                  3: random_guassian_SynFun(j0=20,p=0.6),
#              })

j0 = 1
sigma = 3
p = 0.3
coef = 500
SynapseGroup(net=net, src=exc_group, dst=inh_group, behavior={3: Init_synapse(j0=j0, coef=coef, sigma=sigma, p=p)}, tag="GLUTAMATE")
SynapseGroup(net=net, src=exc_group, dst=exc_group, behavior={3: Init_synapse(j0=j0, coef=coef, sigma=sigma, p=p)}, tag="GLUTAMATE_Inner")
SynapseGroup(net=net, src=inh_group, dst=exc_group, behavior={3: Init_synapse(j0=j0, coef=coef, sigma=sigma, p=p)}, tag="GABA")
SynapseGroup(net=net, src=inh_group, dst=inh_group, behavior={3: Init_synapse(j0=j0, coef=coef, sigma=sigma, p=p)}, tag="GABA_Inner")


net.initialize()


net.simulate_iterations(iteration_numbers)


counts1 = np.bincount(net["ng1_evrec",0]["spike",0][:,0], minlength=iteration_numbers+1)
counts2 = np.bincount(net["ng2_evrec",0]["spike",0][:,0], minlength=iteration_numbers+1)


pop_activities = counts1 + counts2
pop_activities = pop_activities/(ng1_size + ng2_size)

# print(net["ng1_rec", 0]["v", 0])
# print(net["ng1_evrec", 0]["spike", 0])

font1 = {'size':17}
figure_size = (20,6)

# plt.figure(figsize=figure_size)
# plt.plot(net["ng1_rec", 0].variables["v"][:,:3])
# plt.title("Membrane Potential", fontdict=font1)
# plt.xlabel("Time", fontdict=font1)
# plt.ylabel("u(t)", fontdict=font1)
# plt.show()

# plt.figure(figsize=figure_size)
# plt.plot(net["ng2_rec", 0].variables["v"][:])
# plt.title("Membrane Potential", fontdict=font1)
# plt.xlabel("Time", fontdict=font1)
# plt.ylabel("u(t)", fontdict=font1)
# plt.show()

plt.figure(figsize=figure_size)
plt.scatter(net["ng1_evrec",0]["spike",0][:,0],net["ng1_evrec",0]["spike",0][:,1])
plt.scatter(net["ng2_evrec",0]["spike",0][:,0],net["ng2_evrec",0]["spike",0][:,1]+ng1_size+5)
plt.title("Spikes", fontdict=font1)
plt.xlabel("Time", fontdict=font1)
plt.ylabel("Neuron", fontdict=font1)
plt.legend(["NG1" , "NG2"], loc = "lower left")
plt.xlim(left=0)
plt.show()

#
# plt.figure(figsize=(8,6))
# plt.plot(net["ng2_rec", 0]["v", 0][:,:3])
# plt.title("Membrane Potential", fontdict=font1)
# plt.xlabel("Time", fontdict=font1)
# plt.ylabel("u(t)", fontdict=font1)
# plt.show()

#

# plt.figure(figsize=figure_size)
# plt.plot(net["I",0])
# # plt.ylim(0, 40)
# plt.title("Input current for 10 destination neurons", fontdict=font1)
# plt.xlabel("Time", fontdict=font1)
# plt.ylabel("I(t)", fontdict=font1)
# plt.show()


plt.figure(figsize=figure_size)
plt.plot(net["I_P",0])
plt.ylim(0, 60)
plt.title("Input current", fontdict=font1)
plt.xlabel("Time", fontdict=font1)
plt.ylabel("I(t)", fontdict=font1)
plt.show()


plt.figure(figsize=figure_size)
plt.plot(pop_activities)
# plt.ylim(0, 40)
plt.title("Population Activities", fontdict=font1)
plt.xlabel("Time", fontdict=font1)
plt.ylabel("I(t)", fontdict=font1)
plt.show()


# plt.figure(figsize=figure_size)
# plt.plot(net["I",1])
# plt.ylim(0, 70)
# plt.title("Current", fontdict=font1)
# plt.xlabel("Time", fontdict=font1)
# plt.ylabel("I(t)", fontdict=font1)
# plt.show()
# #
# plt.scatter(net["ng2_rec", 0]["spike", 0][:,0], net["ng2_rec", 0]["spike", 0][:,1])
# plt.title("Spike", fontdict=font1)
# plt.xlabel("Time", fontdict=font1)
# plt.ylabel("Neuron", fontdict=font1)
# plt.show()

#
# plt.figure(figsize=(18,6))
# plt.plot(net["first_term", 0])
# plt.title("first_term", fontdict=font1)
# plt.xlabel("Time", fontdict=font1)
# plt.ylabel("I(t)", fontdict=font1)
# plt.show()
#
#
#
#
# plt.figure(figsize=(18,6))
# plt.plot(net["w", 0], label="W")
# plt.plot(net["first_term", 0], label="first_term")
# plt.title("W", fontdict=font1)
# # plt.title("v", fontdict=font1)
# plt.xlabel("Time", fontdict=font1)
# # plt.ylabel("I(t)", fontdict=font1)
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(18,6))
# plt.plot(net["w", 0], label="W")
# plt.plot(net["v", 0], label="V")
# plt.title("W", fontdict=font1)
# # plt.title("v", fontdict=font1)
# plt.xlabel("Time", fontdict=font1)
# # plt.ylabel("I(t)", fontdict=font1)
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(18,6))
# plt.plot(net["third_term", 0])
# plt.title("third_term", fontdict=font1)
# plt.xlabel("Time", fontdict=font1)
# plt.ylabel("I(t)", fontdict=font1)
# plt.show()