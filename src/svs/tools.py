#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common tools shared by submodules
"""

import sys
import utility


def submit_or_skip(prog, args, positional, optional, day=12, log='%x.%j.log'):
    if args.submit or args.hold:
        wd = utility.make_directory(args.wd, task=args.task, status=-1)
        activate = utility.check_executable(prog, task=args.task, status=-2).strip().replace(prog, 'activate')
        venv = f'source {activate}\ncd {wd}'
        cmdline = utility.format_cmd(prog, args, positional, optional)
        
        cpu = args.cpu if hasattr(args, 'cpu') else 1
        gpu = args.gpu if hasattr(args, 'gpu') else 0
        return_code, job_id = utility.submit(cmdline, venv=venv, cpu=cpu, gpu=gpu, name=prog, day=day,
                                             hold=args.hold, script=f'{prog}.sh')
        utility.task_update(return_code, job_id, args.task, 0)
        sys.exit(0)
