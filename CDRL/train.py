import argparse
import os
import sys
import time
current_time = time.strftime("%Y%m%d_%H-%M")
parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-w', '--num-workers', default=1, type=int,
                    help="Number of workers")
parser.add_argument('-m', '--num-MTL-workers', nargs='+', type=int,
                    help="Number of multi task workers, e.g. '3 3' 3 workers for 1st task, 2 workers for second tasks")
parser.add_argument('-r', '--remotes', default=None,
                    help='The address of pre-existing VNC servers and '
                         'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901).')
parser.add_argument('-e', '--env-id', type=str, default="PongDeterministic-v3_Bowling-v0",
                    help="Environment id")
parser.add_argument('-l', '--log-dir', type=str, default="/tmp/cdrl/"+current_time,
                    help="Log directory path")


def new_tmux_cmd(session, name, cmd):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(str(v) for v in cmd)
    return name, "tmux send-keys -t {}:{} '{}' Enter".format(session, name, cmd)


def create_tmux_commands(session, num_workers, remotes, env_id, logdir, shell='sh'):
    # for launching the TF workers and for launching tensorboard
    base_cmd = [
        'CUDA_VISIBLE_DEVICES=', sys.executable, 'worker.py',
        '--log-dir', logdir, '--env-id', env_id,
        '--num-workers', str(num_workers)]

    if remotes is None:
        remotes = ["1"] * num_workers
    else:
        remotes = remotes.split(',')
        assert len(remotes) == num_workers

    cmds_map = [new_tmux_cmd(session, "ps", base_cmd + ["--job-name", "ps"])]
    for i in range(num_workers):
        cmds_map += [new_tmux_cmd(session,
            "w-%d" % i, base_cmd + ["--job-name", "worker", "--task", str(i), "--remotes", remotes[i]])]

    cmds_map += [new_tmux_cmd(session, "tb", ["tensorboard --logdir {} --port 12345".format(logdir)])]
    cmds_map += [new_tmux_cmd(session, "htop", ["htop"])]

    windows = [v[0] for v in cmds_map]

    cmds = [
        "mkdir -p {}".format(logdir),
        "tmux kill-session -t {}".format(session),
        "tmux new-session -s {} -n {} -d {}".format(session, windows[0], shell)
    ]
    for w in windows[1:]:
        cmds += ["tmux new-window -t {} -n {} {}".format(session, w, shell)]
    cmds += ["sleep 1"]
    for window, cmd in cmds_map:
        cmds += [cmd]

    return cmds


def run():
    args = parser.parse_args()

    cmds = create_tmux_commands("a3c", args.num_workers, args.remotes, args.env_id, args.log_dir)
    print("\n".join(cmds))
    os.system("\n".join(cmds))


def new_nohup_cmd(session, name, cmd, logdir):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(str(v) for v in cmd)
    return "CUDA_VISIBLE_DEVICES= nohup {} > {}/nohups/{}.nohupout 2>&1 &".format(cmd, logdir, name + "_" + session)


def create_nohup_commands(session, args, shell='sh'):
    args.num_workers = sum(args.num_MTL_workers)
    num_workers = args.num_workers
    remotes = args.remotes
    env_id = args.env_id
    logdir = args.log_dir


    # for launching the TF workers and for launching tensorboard
    base_cmd = [sys.executable, "-u", 'worker.py',
                '--log-dir', logdir, '--env-id', env_id,
                '--num-workers', str(num_workers)]

    if remotes is None:
        remotes = ["1"] * num_workers
    else:
        remotes = remotes.split(',')
        assert len(remotes) == num_workers

    cmds_map = [new_nohup_cmd(session, "ps", base_cmd + ["--job-name", "ps"], logdir), "ps=$!"] # command for parameter server

    num_mtl_workers = args.num_MTL_workers  # how many workers for each task are we running
    num_mtl_tasks = len(num_mtl_workers) # how many tasks are we running
    assert num_workers == sum(num_mtl_workers)
    past = 0
    for kk in range(num_mtl_tasks):
        for i in range(num_mtl_workers[kk]):
            cmds_map += [new_nohup_cmd(session,
                    "w-%d-task%d" % (past, kk), base_cmd + ["--job-name", "worker", "--worker-id", str(past), "--target-task", str(kk),
                                            "--remotes", remotes[i]], logdir),
                    "w{}=$!".format(str(past))] #, "sleep 1"]
            past += 1
            if past == 0:
                cmds_map += ["sleep 1"]

    cmds = [
        "mkdir -p {}".format(logdir),
        "mkdir -p {}/nohups".format(logdir),
    ]
    cmds += ["sleep 1"]

    for cmd in cmds_map:
        cmds += [cmd]

    cmds += ['echo \'kill -9 \'' + "$ps " + " ".join(["$w"+str(i) for i in range(num_workers)]) + " > " + logdir + "/kill_process.sh"]
    return cmds

def record_command(cmds, fname):
    with open(fname, "w") as f:
        f.write("\n".join(cmds))

def runnohup():
    args = parser.parse_args()
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


    fname = args.log_dir + "/run.sh"
    cmds = create_nohup_commands("a3c", args)
    print("\n".join(cmds))
    record_command(cmds, fname)
    os.system("sh " + fname)
    print "remember to check pid with: ps -ef |grep python | grep linkaixi"

if __name__ == "__main__":
    runnohup()

    # python train.py --num-MTL-workers 8 8 --env-id PongDeterministic-v3_Bowling-v0
