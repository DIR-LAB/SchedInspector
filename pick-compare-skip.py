import time
import joblib
import os
import os.path as osp
import tensorflow as tf
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph

import gym
from gym import spaces
from gym.spaces import Box, Discrete
from gym.utils import seeding

import random
import math
import numpy as np
import sys

from statistics import mean

from HPCSimSkip import *

import matplotlib.pyplot as plt
plt.rcdefaults()
tf.enable_eager_execution()

def load_policy(model_path, itr='last'):
    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(model_path) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(model_path, 'simple_save'+itr))

    # get the correct op for executing actions
    pi = model['pi']
    v = model['v']
    out = model['out']
    get_out = lambda x ,y  : sess.run(out, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES), model['mask']:y.reshape(-1, 2)})
    # make function for producing an action given a single state
    get_probs = lambda x ,y  : sess.run(pi, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES), model['mask']:y.reshape(-1,2)})
    get_v = lambda x : sess.run(v, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES)})
    return get_probs, get_out

def action_from_obs(o):
    lst = []
    for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
        if o[i] == 0 and o[i + 1] == 1 and o[i + 2] == 1 and o[i + 3] == 0:
            pass
        elif o[i] == 1 and o[i + 1] == 1 and o[i + 2] == 1 and o[i + 3] == 1:
            pass
        else:
            lst.append((o[i+1],math.floor(i/JOB_FEATURES)))
    min_time = min([i[0] for i in lst])
    result = [i[1] for i in lst if i[0]==min_time]
    return result[0]

#@profile
def run_policy(env, get_probs, get_out, nums, iters, score_type):
    rl_r = []
    f1_r = [] 
    f2_r = []
    sjf_r = []
    #small_r = []
    wfp_r = []
    uni_r = []

    fcfs_r = []
    # time_total = 0
    # num_total = 0
    #ff = open("feature_set2",'w')
    for iter_num in range(0, iters):
        #start = iter_num *args.len
        #start = iter_num
        start = env.np_random.randint(0, env.loads.size() - 256)
        env.reset_for_test(nums,start)
        schedule_algos = [env.fcfs_score, env.lcfs_score, env.smallest_score, env.largest_score, env.sjf_score, env.lpf_score, env.saf_score,
                env.laf_score, env.sexp_score, env.lexp_score, env.srf_score, env.lrf_score, env.multifactor_score, env.f1_score]

        # f1_r.append(env.score_acorss_users(env.per_user_scores(env.schedule_curr_sequence_reset(env.f1_score)).values()))
        # f2_r.append(env.score_acorss_users(env.per_user_scores(env.schedule_curr_sequence_reset(env.f2_score)).values()))
        # f2_r.append(sum(env.schedule_curr_sequence_reset(env.f2_score).values()))
        # uni_r.append(env.score_acorss_users(env.per_user_scores(env.schedule_curr_sequence_reset(env.uni_score)).values()))
        # wfp_r.append(env.score_acorss_users(env.per_user_scores(env.schedule_curr_sequence_reset(env.wfp_score)).values()))
        schedule_logs = env.schedule_curr_sequence_reset(schedule_algos[args.sched_algo])
        sjf_r.append(-sum(schedule_logs.values()))
        # small_r.append(env.score_acorss_users(env.per_user_scores(env.schedule_curr_sequence_reset(env.smallest_score)).values()))
        # fcfs_r.append(env.score_acorss_users(env.per_user_scores(env.schedule_curr_sequence_reset(env.fcfs_score)).values()))

        o = env.build_observation()
        print ("schedule: ", end="")
        rl = 0
        total_decisions = 0
        skip_decisions = 0
        while True:
            skip_ = []
            lst = [1, 1]
            for i in range(0, MAX_QUEUE_SIZE*JOB_FEATURES, JOB_FEATURES):
                job = o[i:i+JOB_FEATURES]
                if job[-2] == 1:
                    lst = [1, 0]

            out = get_out(o,np.array(lst))
            softmax_out = tf.nn.softmax(out)
            confidence = tf.reduce_max(softmax_out)
            total_decisions += 1.0
            if confidence > 0:
                # start_time = time.time()
                pi = get_probs(o, np.array(lst))
                # time_total += time.time() - start_time
                # num_total += 1
                # print(start_time, time_total, num_total)
                a = pi[0]
            else:
                # print('SJF')
                a = 0
                #a = action_from_obs(o)
            # print(out)
            # v_t = get_value(o)
            #ff.write("{}\t{}\t{}\n".format("\t".join(map(str, o)), softmax_out[0][0], softmax_out[0][1]))

            if a == 1:
                skip_decisions += 1
            print(str(a), end="|")
            o, r, d, _ = env.step_for_test(a)
            rl += r
            if d:
                # print("RL decision ratio:",rl_decisions/total_decisions)
                print("Sequence Length:",total_decisions, "Skip ratio:", skip_decisions/total_decisions)
                break
        #with open("rl_logs", 'w') as f:
        #    for i in sorted(env.loads[:50000]):
        #        f.write("{}\t{}\t{}\t{}\n".format(i.job_id, i.submit_time, i.run_time, i.scheduled_time))
        rl_r.append(-rl)
        print ("")

    # plot
    all_data = []
    # all_data.append(fcfs_r)
    # all_data.append(wfp_r)
    # all_data.append(uni_r)
    all_data.append(sjf_r)
    # all_data.append(f1_r)
    all_data.append(rl_r)
    #all_data.append(fcfs_r)
    print("Mean SJF:", mean(sjf_r))
    print("Mean RL:", mean(rl_r))
    print("%:", (mean(sjf_r) - mean(rl_r))/mean(sjf_r))
    #ff.close()    

    all_medians = []
    for p in all_data:
        all_medians.append(np.median(p))

    # plt.rc("font", size=45)
    # plt.figure(figsize=(12, 7))
    plt.rc("font", size=33)
    plt.figure(figsize=(5, 7))
    axes = plt.axes()

    xticks = [y + 1 for y in range(len(all_data))]
    plt.plot(xticks[0:1], all_data[0:1], 'o', linewidth=1, color='black')
    plt.plot(xticks[1:2], all_data[1:2], 'o', linewidth=1, color='black')
    # plt.plot(xticks[2:3], all_data[2:3], 'o', color='darkorange')
    # plt.plot(xticks[3:4], all_data[3:4], 'o', color='darkorange')
    # plt.plot(xticks[4:5], all_data[4:5], 'o', color='darkorange')
    # plt.plot(xticks[5:6], all_data[5:6], 'o', color='darkorange')
    #plt.plot(xticks[6:7], all_data[6:7], 'o', color='darkorange')

    plt.boxplot(all_data, showfliers=False, meanline=True, showmeans=True, widths=0.5, medianprops={"linewidth":0},meanprops={"color":"black", "linewidth":4,"linestyle":"solid"})

    #plt.ylim([0,600])

    #axes.yaxis.grid(True)
    axes.set_xticks([y + 1 for y in range(len(all_data))])
    xticklabels = ['Original', 'Delayed']
    # xticklabels = ['FCFS', 'WFP', 'UNI', 'SJF', 'F1', 'RL']
    # xticklabels = ['FCFS', 'WFP', 'UNI', 'SJF', 'RL']
    plt.setp(axes, xticks=[y + 1 for y in range(len(all_data))],
             xticklabels=xticklabels)
    
    #if score_type == 0:
    #    plt.ylabel("Average bounded slowdown")
    #elif score_type == 1:
    #    plt.ylabel("Average waiting time")
    #elif score_type == 2:
    #    plt.ylabel("Average turnaround time")
    #elif score_type == 3:
    #    plt.ylabel("Resource utilization")
    #else:
    #    raise NotImplementedError

    # plt.ylabel("Average waiting time (s)")
    #plt.xlabel("Scheduling Policies")
    # plt.tick_params(axis='both', which='major', labelsize=40)
    # plt.tick_params(axis='both', which='minor', labelsize=40)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.tick_params(axis='both', which='minor', labelsize=30)
    plt.tight_layout(pad=0.5)

    plt.savefig("temp.png")
    plt.show()

if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--rlmodel', type=str, default="./data/logs/ppo/ppo_s0")
    parser.add_argument('--workload', type=str, default='./data/lublin_256.swf')
    parser.add_argument('--len', '-l', type=int, default=256)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--iter', '-i', type=int, default=10)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--job_score_type', type=int, default=0)
    parser.add_argument('--user_score_type', type=int, default=0)
    parser.add_argument('--batch_job_slice', type=int, default=0)
    parser.add_argument('--sched_algo', type=int, default=4)

    args = parser.parse_args()

    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    model_file = os.path.join(current_dir, args.rlmodel)

    get_probs, get_value = load_policy(model_file, 'last') 
    
    # initialize the environment from scratch
    env = HPCEnvSkip(shuffle=args.shuffle, backfil=args.backfil, skip=args.skip, job_score_type=args.job_score_type,
                 batch_job_slice=args.batch_job_slice, build_sjf=False, sched_algo=args.sched_algo)
    env.my_init(workload_file=workload_file)
    env.seed(args.seed)

    start = time.time()
    run_policy(env, get_probs, get_value, args.len, args.iter, args.job_score_type)
    print("time elapse: {}".format(time.time()-start))
