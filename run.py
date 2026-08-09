import importlib

from common.args import run_args

if __name__ == "__main__":
    task = importlib.import_module(run_args.module)
    getattr(task, run_args.action)()
