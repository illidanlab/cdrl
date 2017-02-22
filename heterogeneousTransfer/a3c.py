from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
from model import LSTMPolicy, logitsProj
import six.moves.queue as queue
import scipy.signal
import threading
from collections import defaultdict

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0):
    """
given a rollout, compute its returns and the advantage
"""
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]
    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])

class PartialRollout(object):
    """
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []

    def add(self, state, action, reward, value, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)

class RunnerThread(threading.Thread):
    """
One of the key distinctions between a normal environment and a universe environment
is that a universe environment is _real time_.  This means that there should be a thread
that would constantly interact with the environment and tell it what to do.  This thread is here.
"""
    def __init__(self, env, policy, num_local_steps, name):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.name = name

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()


    def run(self):

        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.env, self.policy, self.num_local_steps, self.summary_writer, self.name)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.
            self.queue.put(next(rollout_provider), timeout=600.0)
            # lkx: rollout_provider is generator, to generate rollouts once it is computed
            # Generators are iterators,
            # but you can only iterate over them once.
            # It's because they do not store all the values in memory,
            # they generate the values on the fly


def env_runner(env, policy, num_local_steps, summary_writer, name):
    """
The logic of the thread runner.  In brief, it constantly keeps on running
the policy, and as long as the rollout exceeds a certain length, the thread 
runner appends the policy to the queue.
"""
    last_state = env.reset()
    last_features = policy.get_initial_features()
    length = 0
    rewards = 0

    while True:
        terminal_end = False
        rollout = PartialRollout()

        for _ in range(num_local_steps):
            fetched = policy.act(last_state, *last_features)
            action, value_, features = fetched[0], fetched[1], fetched[2:]
            # argmax to convert from one-hot
            state, reward, terminal, info = env.step(action.argmax())

            # collect the experience
            rollout.add(last_state, action, reward, value_, terminal, last_features)
            length += 1
            rewards += reward

            last_state = state
            last_features = features
            # info: {'global/episode_reward': -21.0, 'global/reward_per_time': -31.19907423314122, 'global/episode_time': 0.6730968952178955, 'global/episode_length': 764}
            if info:
                summary = tf.Summary()
                for k, v in info.items(): #
                    summary.value.add(tag=k+name, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if terminal or length >= timestep_limit:
                terminal_end = True
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                last_features = policy.get_initial_features()
                print(name + " Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
                length = 0
                rewards = 0
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout

class A3C(object):
    def __init__(self, envs, workerid, target_task):
        """
An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""

        self.env = envs
        self.num_tasks = num_tasks = 2
        self.target_task = target_task = 1
        self.aux_tasks_id = 0
        self.workerid = workerid
        self.network = [None] * self.num_tasks
        self.local_network = [None] * self.num_tasks
        self.global_step = [None] * self.num_tasks

        pi = [None] * self.num_tasks
        worker_device = "/job:worker/task:{}/cpu:0".format(workerid)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global" + str(0)):
                self.network[0] = LSTMPolicy(envs[0].observation_space.shape, envs[0].action_space.n)
                self.global_step[0] = tf.get_variable("global_step", [], tf.int32, initializer=tf.zeros_initializer,
                                                      trainable=False)
            with tf.variable_scope("global" + str(1)):
                self.network[1] = LSTMPolicy(envs[1].observation_space.shape, envs[1].action_space.n)
                self.global_step[1] = tf.get_variable("global_step", [], tf.int32, initializer=tf.zeros_initializer,
                                                      trainable=False)
            with tf.variable_scope("globallogits"):  # network for projection logits.
                self.logitProjnet = logitsProj(envs[0].action_space.n, envs[1].action_space.n)

        with tf.device(worker_device):
            with tf.variable_scope("local" + str(0)):
                self.local_network[0] = pi[0] = LSTMPolicy(envs[0].observation_space.shape, envs[0].action_space.n)
                pi[0].global_step = self.global_step[0]

            with tf.variable_scope("local" + str(1)):
                self.local_network[1] = pi[1] = LSTMPolicy(envs[1].observation_space.shape, envs[1].action_space.n)
                pi[1].global_step = self.global_step[1]

            with tf.variable_scope("local" + "logits"):
                self.local_logitProjnet = logitsProj(envs[0].action_space.n, envs[1].action_space.n)


            self.ac = [None] * num_tasks
            self.adv = [None] * num_tasks
            self.r = [None] * num_tasks
            log_prob_tf = [None] * num_tasks
            prob_tf = [None] * num_tasks
            pi_loss = [None] * num_tasks
            vf_loss = [None] * num_tasks
            entropy = [None] * num_tasks
            bs = [None] * num_tasks
            self.loss = [None] * num_tasks
            self.runner = [None] * num_tasks
            grads = [None] * num_tasks
            self.summary_op = [[None, None] for i in np.arange(num_tasks)]  #[[None, None], [None, None]] 2 tasks
            self.sync = [None] * num_tasks
            grads_and_vars = [None] * num_tasks
            self.inc_step = [None] * num_tasks
            opt = [None] * num_tasks
            self.train_op = [None] * num_tasks
            self.target_logits = [None] * num_tasks
            soft_p_temperature = [None] * num_tasks
            soft_t_temperature = [None] * num_tasks
            self.KD_trainop = [None] * num_tasks
            kl_loss = [None] * num_tasks
            grads_kd = [None] * num_tasks
            grads_and_vars_kd = [None] * num_tasks
            optkd = [None] * num_tasks
            for ii in np.arange(num_tasks):
            # start to build loss for target network
                self.ac[ii] = tf.placeholder(tf.float32, [None, envs[ii].action_space.n], name="ac"+str(ii))
                self.adv[ii] = tf.placeholder(tf.float32, [None], name="adv"+str(ii))
                self.r[ii] = tf.placeholder(tf.float32, [None], name="r"+str(ii))

                log_prob_tf[ii] = tf.nn.log_softmax(pi[ii].logits)
                prob_tf[ii] = tf.nn.softmax(pi[ii].logits)

                # the "policy gradients" loss:  its derivative is precisely the policy gradient
                # notice that self.ac is a placeholder that is provided externally.
                # ac will contain the advantages, as calculated in process_rollout
                pi_loss[ii] = - tf.reduce_sum(tf.reduce_sum(log_prob_tf[ii] * self.ac[ii], [1]) * self.adv[ii])

                # loss of value function
                vf_loss[ii] = 0.5 * tf.reduce_sum(tf.square(pi[ii].vf - self.r[ii]))
                entropy[ii] = - tf.reduce_sum(prob_tf[ii] * log_prob_tf[ii])

                bs[ii] = tf.to_float(tf.shape(pi[ii].x)[0])
                self.loss[ii] = pi_loss[ii] + 0.5 * vf_loss[ii] - entropy[ii] * 0.01

                # 20 represents the number of "local steps":  the number of timesteps
                # we run the policy before we update the parameters.
                # The larger local steps is, the lower is the variance in our policy gradients estimate
                # on the one hand;  but on the other hand, we get less frequent parameter updates, which
                # slows down learning.  In this code, we found that making local steps be much
                # smaller than 20 makes the algorithm more difficult to tune and to get to work.
                name = "task" + str(ii)
                self.runner[ii] = RunnerThread(envs[ii], pi[ii], 20, name)

                grads[ii] = tf.gradients(self.loss[ii], pi[ii].var_list)
                summaries1 = list() # summary when it's target tasks
                summaries1.append(tf.scalar_summary("model/policy_loss"+str(ii), pi_loss[ii] / bs[ii]))
                summaries1.append(tf.scalar_summary("model/value_loss"+str(ii), vf_loss[ii] / bs[ii]))
                summaries1.append(tf.scalar_summary("model/entropy"+str(ii), entropy[ii] / bs[ii]))
                summaries1.append(tf.image_summary("model/state"+str(ii), pi[ii].x))
                summaries1.append(tf.scalar_summary("model/grad_global_norm"+str(ii), tf.global_norm(grads[ii])))
                summaries1.append(tf.scalar_summary("model/var_global_norm"+str(ii), tf.global_norm(pi[ii].var_list)))
                summaries1.append(tf.histogram_summary("model/action_weight"+str(ii), prob_tf[ii]))

                summaries2 = list() # summary when it's aux tasks.
                summaries2.append(tf.histogram_summary("model/action_weight" + str(ii), prob_tf[ii]))
                summaries2.append(tf.scalar_summary("model/entropy" + str(ii), entropy[ii] / bs[ii]))
                self.summary_op[ii][0] = tf.merge_summary(summaries1)
                self.summary_op[ii][1] = tf.merge_summary(summaries2)

                grads[ii], _ = tf.clip_by_global_norm(grads[ii], 40.0)


                zipvars = zip(pi[ii].var_list, self.network[ii].var_list)
                self.sync[ii] = tf.group(*[v1.assign(v2) for v1, v2 in zipvars])

                grads_and_vars[ii] = list(zip(grads[ii], self.network[ii].var_list))
                self.inc_step[ii] = self.global_step[ii].assign_add(tf.shape(pi[ii].x)[0])

                # each worker has a different set of adam optimizer parameters
                opt[ii] = tf.train.AdamOptimizer(1e-4)
                self.train_op[ii] = tf.group(opt[ii].apply_gradients(grads_and_vars[ii]), self.inc_step[ii])

                # knowledge distillation
                self.target_logits[ii] = tf.placeholder(tf.float32, [None, envs[ii].action_space.n], name="target_logits")  # logits from teacher
                Tao = 1.0  # temperature used for distillation.
                soft_p_temperature[ii] = tf.nn.softmax(pi[ii].logits_fordistill)

                soft_t_temperature[ii] = tf.nn.softmax(tf.truediv(self.target_logits[ii], Tao))

                kl_loss[ii] = tf.reduce_mean(tf.reduce_sum(
                    soft_t_temperature[ii] * tf.log(1e-12 + tf.truediv(soft_t_temperature[ii], soft_p_temperature[ii])), 1))

                grads_kd[ii] = tf.gradients(kl_loss[ii], pi[ii].var_list)
                grads_kd[ii], _ = tf.clip_by_global_norm(grads_kd[ii], 40.0)
                grads_and_vars_kd[ii] = list(zip(grads_kd[ii], self.network[ii].var_list))
                optkd[ii] = tf.train.AdamOptimizer(1e-4)
                self.KD_trainop[ii] = optkd[ii].apply_gradients(grads_and_vars_kd[ii])


            'learning logits projection'
            zipvars = zip(self.local_logitProjnet.var_list, self.logitProjnet.var_list)
            self.sync_logits = tf.group(*[v1.assign(v2) for v1, v2 in zipvars])
            # soft_student_logits = tf.nn.softmax(pi[target_task].logits)
            self.logits_stu = tf.placeholder(tf.float32, [None, envs[1].action_space.n])
            soft_student_logits = tf.nn.softmax(self.logits_stu)
            soft_teacher_logits = tf.nn.softmax(self.local_logitProjnet.logits_out)
            self.proj_loss = proj_loss = tf.reduce_mean(tf.reduce_sum(
                        soft_teacher_logits * tf.log(1e-12 + tf.truediv(soft_teacher_logits, soft_student_logits)), 1))  # target task --> student
            grad_logproj = tf.gradients(proj_loss, self.local_logitProjnet.var_list)
            grad_logproj, _ = tf.clip_by_global_norm(grad_logproj, 40.0)
            grads_and_vars_logproj = list(zip(grad_logproj, self.logitProjnet.var_list))
            optlgproj = tf.train.AdamOptimizer(1e-4)
            self.lgproj_trainop = optlgproj.apply_gradients(grads_and_vars_logproj)
            summaries_proj = list()
            summaries_proj.append(tf.scalar_summary("model/proj_loss", self.proj_loss))
            summaries_proj.append(tf.histogram_summary("model/proj_student", soft_student_logits))
            summaries_proj.append(tf.histogram_summary("model/proj_teacher", soft_teacher_logits))
            self.summary_op_proj = tf.merge_summary(summaries_proj)

            self.summary_writer = None
            self.local_steps = 0
            self.replay_buffer = []

    def start(self, sess, summary_writer):


        for ii in np.arange(self.num_tasks):

            self.runner[ii].start_runner(sess, summary_writer)

        self.summary_writer = summary_writer


    def pull_batch_from_queue(self):
        """
self explanatory:  take a rollout from the queue of the thread runner.
"""
        rollout = self.runner[self.target_task].queue.get(timeout=600.0)  # need to pull from queue so that it can continue
        # for ii in self.aux_tasks_id:
        rollout_aux = self.runner[self.aux_tasks_id].queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner[self.target_task].queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """
process grabs a rollout that's been produced by the thread runner,
and updates the parameters.  The update is then sent to the parameter
server. 
"""
        # for ii in np.arange(self.num_tasks):
        # sess.run(self.sync[self.target_task])  # copy weights from shared to local
        target_task = self.target_task
        for ii in np.arange(self.num_tasks):
            sess.run(self.sync[ii])  # copy weights from shared to local
        sess.run(self.sync_logits)

        rollout = self.pull_batch_from_queue()

        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)

        # should_compute_summary = self.workerid == 0 and self.local_steps % 11 == 0
        should_compute_summary = self.local_steps % 11 == 0

        if should_compute_summary:
            fetches = [self.summary_op[target_task][0], self.train_op[target_task], self.local_network[target_task].logits,
                       self.global_step[target_task]]
        else:
            fetches = [self.train_op[target_task], self.local_network[self.target_task].logits,
                       self.global_step[target_task]]

        feed_dict = {
            self.local_network[target_task].x: batch.si,
            self.ac[target_task]: batch.a,
            self.adv[target_task]: batch.adv,
            self.r[target_task]: batch.r,
            self.local_network[target_task].state_in[0]: batch.features[0],
            self.local_network[target_task].state_in[1]: batch.features[1],
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        num_trainnon = 1000000
        if fetched[-1] >= num_trainnon: # and fetched[-1] <= 10000000:

            'distillation knowledge from teacher net to student net.'
            aux_i = self.aux_tasks_id
            feed_dict_logits = {
                self.local_network[aux_i].x: batch.si,
                self.local_network[aux_i].state_in[0]: batch.features[0],
                self.local_network[aux_i].state_in[1]: batch.features[1],
            }
            fetches_logits = [self.local_network[aux_i].logits]
            fetched_logits = sess.run(fetches_logits, feed_dict=feed_dict_logits)


            # 'get non linear mapped logits'
            feed_dict_logproj = {
                self.local_logitProjnet.x: fetched_logits[0],
            }
            featched_mapedlogits = sess.run([self.local_logitProjnet.logits_out], feed_dict=feed_dict_logproj)

            #'training nonlinear logits map'
            if should_compute_summary:
                feaches_proj = [self.lgproj_trainop, self.summary_op_proj]
            else:
                feaches_proj = [self.lgproj_trainop]
            feed_dict_train_logits = {self.local_logitProjnet.x: fetched_logits[0],
                                      self.logits_stu: fetched[-2]}
            featched_proj = sess.run(feaches_proj, feed_dict=feed_dict_train_logits)

        num_distill = 2000000
        # 'distill to student' start to distillation after num_distill millions
        if fetched[-1] >= num_distill : #and fetched[-1] <= 10000000 :
            feed_dict_kd = {
                self.local_network[target_task].x: batch.si,
                self.target_logits[target_task]: featched_mapedlogits[0],
                self.local_network[target_task].state_in[0]: batch.features[0],
                self.local_network[target_task].state_in[1]: batch.features[1],
            }
            sess.run(self.KD_trainop[target_task], feed_dict=feed_dict_kd)


        if should_compute_summary:
            if fetched[-1] >= num_trainnon: # and fetched[-1] <= 10000000:
                self.summary_writer.add_summary(tf.Summary.FromString(featched_proj[-1]), fetched[-1])

            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1

