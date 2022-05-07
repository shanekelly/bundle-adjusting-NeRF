import numpy as np
import os
import sys
import time
import torch
import importlib

import options
from ipdb import launch_ipdb_on_exception
from util import log


def main():

    log.process(os.getpid())
    log.title("[{}] (PyTorch code for evaluating NeRF/BARF)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)

    with torch.cuda.device(opt.device):

        model = importlib.import_module("model.{}".format(opt.model))
        m = model.Model(opt)

        m.load_dataset(opt, eval_split="test")
        m.build_networks(opt)

        if opt.model == "barf" and opt.eval.vid_pose:
            m.generate_videos_pose(opt)

        if opt.data.dataset in ["blender", "llff"]:
            m.restore_checkpoint(opt)
            m.evaluate_full(opt)

        if opt.eval.vid_novel_view:
            m.restore_checkpoint(opt)
            m.generate_videos_synthesis(opt)

        if opt.eval.render_train:
            m.render_train(opt)

        if opt.model == 'barf' and opt.eval.save_pose:
            m.save_pose_TUM(opt)


if __name__ == "__main__":
    with launch_ipdb_on_exception():
        main()
