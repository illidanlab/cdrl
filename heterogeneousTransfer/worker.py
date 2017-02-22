#!/usr/bin/env python
import cv2
import go_vncdriver
import tensorflow as tf
import argparse
import logging
import os
from a3c import A3C, PartialRollout
from envs import create_env
import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)

def run(args, server):
    # lkx: client and remote doesn't mater for non VNC and flash game
    # env = create_env(args.env_id, client_id=str(args.task), remotes=args.remotes)
    # trainer = A3C(env, args.task)


    target_task = 1   # int(args.target_task)
    env_names = args.env_id.split("_")
    envs = [create_env(env_name, client_id=str(args.worker_id), remotes=args.remotes) for env_name in env_names]

    trainer = A3C(envs, int(args.worker_id), target_task)

    # Variable names that start with "local" are not saved in checkpoints.
    variables_to_save = [v for v in tf.all_variables() if not v.name.startswith("local")]
    init_op = tf.initialize_variables(variables_to_save)
    init_all_op = tf.initialize_all_variables()
    saver = FastSaver(variables_to_save)

    variables_to_restore = [v for v in tf.all_variables() if v.name.startswith("global0")
                            and "global_step" not in v.name] # Adam_2 and 3 cost by the distillation train op
    pre_train_saver = FastSaver(variables_to_restore)


    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)
        pre_train_saver.restore(ses,
                                "../model/model.ckpt-4986751")

    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.worker_id)]) # refer to worker id
    logdir = os.path.join(args.log_dir, 'train')
    summary_writer = tf.train.SummaryWriter(logdir + "_%d" % args.worker_id)
    logger.info("Events directory: %s_%s", logdir, args.worker_id)
    sv = tf.train.Supervisor(is_chief=(args.worker_id == 0),
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,  # Defaults to an Operation that initializes all variables
                             init_fn=init_fn,  # Called after the optional init_op is called
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),  # list the names of uninitialized variables.
                             global_step=trainer.global_step[target_task],
                             save_model_secs=30,
                             save_summaries_secs=30)

    num_taskss = len(envs)

    num_global_steps = 20000000 #10000000

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")
    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        for ii in np.arange(num_taskss):
            sess.run(trainer.sync[ii])
        sess.run(trainer.sync_logits)
        trainer.start(sess, summary_writer)
        global_step = sess.run(trainer.global_step[target_task])
        logger.info("Starting training at step=%d", global_step)
        while not sv.should_stop() and (not num_global_steps or global_step < num_global_steps):
            # if global_step <= 1000000 and np.random.uniform(0, 1) > 0.5:   # todo annealing
            #     batch_aux = trainer.get_knowledge(sess)
            #     trainer.process(sess, batch_aux)
            trainer.process(sess)
            global_step = sess.run(trainer.global_step[target_task])

    # Ask for all the services to stop.
    sv.stop()
    logger.info('reached %s steps. worker stopped.', global_step)

def knowledge_distill(sess, aux_tasks):
    rollout = PartialRollout()
    for aux in aux_tasks:
        rollout.extend(aux.get_knowledge(sess))

    return rollout

def cluster_spec(num_workers, num_ps):
    """
More tensorflow setup for data parallelism
"""
    cluster = {}
    port = 12322

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster

def main(_):
    """
Setting up Tensorflow for data parallel work
"""

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--worker-id', default=0, type=int, help='Task index: is the index of workers')
    parser.add_argument('--job-name', default="worker", help='worker or ps')
    parser.add_argument('--num-workers', default=6, type=int, help='Number of workers')
    parser.add_argument('--log-dir', default="/tmp/pong", help='Log directory path')
    # parser.add_argument('--env-id', default="PongDeterministic-v3", help='Environment id')
    parser.add_argument('--env-id', default="PongDeterministic-v3_PongDeterministic-v3",
                        help='Environment id for multiple tasks')
    parser.add_argument('--target-task', default=0, help='the 0 th tasks/env is the main tasks')
    parser.add_argument('-r', '--remotes', default=None,
                        help='References to environments to create (e.g. -r 20), '
                             'or the address of pre-existing VNC servers and '
                             'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901)')

    args = parser.parse_args()
    spec = cluster_spec(args.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()
    print spec

    if args.job_name == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.worker_id,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=3))
        run(args, server)
    else:
        server = tf.train.Server(cluster, job_name="ps", task_index=args.worker_id,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))

        server.join()

if __name__ == "__main__":
    tf.app.run()
