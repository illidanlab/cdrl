import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.random import normal
import numpy as np


'''one rescource'''
# Dir = "/home/linkaixi/AllData/deepRLTL/20170112_16-02/nohups/"
# Dir = "/home/linkaixi/AllData/deepRLTL/20170112_21-22/nohups/"

# Dir = "/home/linkaixi/AllData/deepRLTL/20170116_10-12/nohups/"

# Dir = "/home/linkaixi/AllData/deepRLTL/20170116_10-21/nohups/"
# data_flag = '0116_10-21'

Dir = "/home/linkaixi/AllData/deepRLTL/20170116_21-10/nohups/"
data_flag = "0116_21-10"

for workerid in range(8):
    # fname = Dir + "w-{}-task{}_a3c.nohupout".format(str(workerid), str(workerid))
    fname = Dir + "w-{}-task0_a3c.nohupout".format(str(workerid))
    worker1task0 = []
    worker1task1 = []
    with open(fname, "r") as f:

        for line in f:
            if "worker"+str(workerid)+"task0 Episode" in line:
                line_list = line.split(" ")
                worker1task0.append(float(line_list[6].replace(".", "")))
            if "worker"+str(workerid)+"task1 Episode" in line:
                line_list = line.split(" ")
                worker1task1.append(float(line_list[6].replace(".", "")))

    worker1task0 = worker1task0[:500]
    worker1task1 = worker1task1[:500]

    fig = plt.figure(0)
    plt.plot(worker1task0, 'r-o', label='worker'+str(workerid)+'task0')
    plt.plot(worker1task1, 'g-*', label='worker'+str(workerid)+'task1')  # colors ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    plt.legend(bbox_to_anchor=(0., 1.01, 1., .101), loc=2,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(Dir + data_flag + 'rewards{}.png'.format(workerid))
    plt.close(fig)




# Dir1 = "/home/linkaixi/AllData/deepRLTL/20170112_16-02/nohups/"
# Dir2 = "/home/linkaixi/AllData/deepRLTL/20170111_10-58/nohups/"
#
#
# distilleda3c = []
# for workerid in [1]:
#     fname = Dir1 + "w-{}-task{}_a3c.nohupout".format(str(workerid), str(workerid))
#
#     with open(fname, "r") as f:
#
#         for line in f:
#             # if "worker"+str(workerid)+"task0 Episode" in line:
#             #     line_list = line.split(" ")
#             #     worker1task0.append(float(line_list[6].replace(".", "")))
#             if "worker"+str(workerid)+"task1 Episode" in line:
#                 line_list = line.split(" ")
#                 distilleda3c.append(float(line_list[6].replace(".", "")))
#
# vanillaa3c = []
# for workerid in [1]:
#         fname = Dir2 + "w-0_a3c.nohupout"
#         worker1task0 = []
#         worker1task1 = []
#         with open(fname, "r") as f:
#
#             for line in f:
#                 if "Episode finished" in line:
#                     line_list = line.split(" ")
#                     vanillaa3c.append(float(line_list[5].replace(".", "")))
#
#
# fig = plt.figure(0)
# plt.plot(distilleda3c, 'r-o', label='distilleda3c')
# plt.plot(vanillaa3c, 'g-*', label='vanillaa3c')  # colors ('b', 'g', 'r', 'c', 'm', 'y', 'k')
# plt.legend(bbox_to_anchor=(0., 1.01, 1., .101), loc=2,
#            ncol=2, mode="expand", borderaxespad=0.)
# plt.savefig(Dir2 + 'rewards_compare2.png')
# plt.close(fig)
