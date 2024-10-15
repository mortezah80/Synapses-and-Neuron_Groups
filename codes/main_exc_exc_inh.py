from pymonntorch import Network, NeuronGroup, Recorder, EventRecorder, SynapseGroup
from lif_exc_inh import LIF
from dt import TimeResolution
from syn import InpSyn_full_fix, InpSyn_full_gussian, InpSyn_random_gussian, SynFun, InpSyn, Init_synapse, full_fix_SynFun, full_guassian_SynFun,random_guassian_SynFun
from current import SetCurrent, SetCurrent_inh
import torch
import numpy as np
from matplotlib import pyplot as plt



torch.manual_seed(73)

iteration_numbers = 1000
exc1_size = 200
exc2_size = 200
inh_size = 40

net = Network(behavior={1: TimeResolution()}, dtype=torch.float64)

exc_group1 = NeuronGroup(
    size=exc1_size,
    net=net,
    behavior={
        2: SetCurrent(value=0),
        # 3: StepFunction(value=20, t0=5, t1=10, t2=20, t3=50, t4=70),
        # 3: StepFunction1(value=5, t0=20, t1=40),
        # 3: StepFunction2(value=220, t0=20, t1=40),
        4: InpSyn_full_fix(),
        5: LIF(R=5,
               threshold=-37,
               u_rest=-67,
               u_back=-75,
               tau=10),
        7: Recorder(tag="exc1_rec", variables=["v"]),
        8: EventRecorder(tag="exc1_evrec", variables=["spike"])})



exc_group2 = NeuronGroup(
    size=exc2_size,
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
        7: Recorder(tag="exc2_rec", variables=["v", "I_P"]),
        8: EventRecorder(tag="exc2_evrec", variables=["spike"])})



inh_group = NeuronGroup(
    size=inh_size,
    net=net,
    behavior={
        2: SetCurrent_inh(value=0),
        # 3: StepFunction(value=20, t0=5, t1=10, t2=20, t3=50, t4=70),
        # 3: StepFunction1(value=5, t0=20, t1=40),
        # 3: StepFunction2(value=220, t0=20, t1=40),
        4: InpSyn_full_gussian(),
        5: LIF(R=5,
               threshold=-37,
               u_rest=-67,
               u_back=-75,
               tau=10),
        7: Recorder(tag="inh_rec", variables=["v", "I"]),
        8: EventRecorder(tag="inh_evrec", variables=["spike"])})

#
# SynapseGroup(net=net, src=ng1, dst=ng2, behavior={
#                  3: random_guassian_SynFun(j0=20,p=0.6),
#              })

j0 = 1
sigma = 3
p = 0.3
coef = 500
SynapseGroup(net=net, src=exc_group1, dst=inh_group, behavior={3: Init_synapse(j0=j0, coef=coef, sigma=sigma, p=p)}, tag="GLUTAMATE")
SynapseGroup(net=net, src=inh_group, dst=exc_group1, behavior={3: Init_synapse(j0=j0, coef=coef, sigma=sigma, p=p)}, tag="GABA")
SynapseGroup(net=net, src=exc_group1, dst=exc_group1, behavior={3: Init_synapse(j0=j0, coef=coef, sigma=sigma, p=p)}, tag="GLUTAMATE_Inner")

SynapseGroup(net=net, src=exc_group2, dst=inh_group, behavior={3: Init_synapse(j0=j0, coef=coef, sigma=sigma, p=p)}, tag="GLUTAMATE")
SynapseGroup(net=net, src=inh_group, dst=exc_group2, behavior={3: Init_synapse(j0=j0, coef=coef, sigma=sigma, p=p)}, tag="GABA")
SynapseGroup(net=net, src=exc_group2, dst=exc_group2, behavior={3: Init_synapse(j0=j0, coef=coef, sigma=sigma, p=p)}, tag="GLUTAMATE_Inner")


# SynapseGroup(net=net, src=inh_group, dst=inh_group, behavior={3: Init_synapse(j0=j0, coef=coef, sigma=sigma, p=p)}, tag="GABA_Inner")


net.initialize()


net.simulate_iterations(iteration_numbers)


counts1 = np.bincount(net["exc1_evrec",0]["spike",0][:,0], minlength=iteration_numbers+1)
counts2 = np.bincount(net["exc2_evrec",0]["spike",0][:,0], minlength=iteration_numbers+1)
counts3 = np.bincount(net["inh_evrec",0]["spike",0][:,0], minlength=iteration_numbers+1)
# counts2 = np.bincount(net["ng2_evrec",0]["spike",0][:,0], minlength=iteration_numbers+1)


pop_activities1 = counts1/exc1_size
pop_activities2 = counts2/exc2_size
pop_activities3 = counts3/inh_size


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
plt.scatter(net["exc1_evrec",0]["spike",0][:,0],net["exc1_evrec",0]["spike",0][:,1])
plt.title("Spikes Exc1", fontdict=font1)
plt.xlabel("Time", fontdict=font1)
plt.xlim(0,iteration_numbers)
plt.ylabel("Neuron", fontdict=font1)
plt.legend(["NG1" , "NG2"], loc = "lower left")
plt.xlim(left=0)
plt.show()

plt.figure(figsize=figure_size)
plt.scatter(net["exc2_evrec",0]["spike",0][:,0],net["exc2_evrec",0]["spike",0][:,1])
plt.title("Spikes Exc2", fontdict=font1)
plt.xlim(0,iteration_numbers)
plt.xlabel("Time", fontdict=font1)
plt.ylabel("Neuron", fontdict=font1)
plt.legend(["NG1" , "NG2"], loc = "lower left")
plt.xlim(left=0)
plt.show()

plt.figure(figsize=figure_size)
plt.scatter(net["inh_evrec",0]["spike",0][:,0],net["inh_evrec",0]["spike",0][:,1])
plt.title("Spikes Inh", fontdict=font1)
plt.xlim(0,iteration_numbers)
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
plt.plot(pop_activities1)
# plt.ylim(0, 40)
plt.title("Population Activities Exc1", fontdict=font1)
plt.xlabel("Time", fontdict=font1)
plt.ylabel("I(t)", fontdict=font1)
plt.show()

plt.figure(figsize=figure_size)
plt.plot(pop_activities2)
# plt.ylim(0, 40)
plt.title("Population Activities Exc2", fontdict=font1)
plt.xlabel("Time", fontdict=font1)
plt.ylabel("I(t)", fontdict=font1)
plt.show()

plt.figure(figsize=figure_size)
plt.plot(pop_activities3)
# plt.ylim(0, 40)
plt.title("Population Activities Inh", fontdict=font1)
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