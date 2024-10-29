from options import Options
from utils import *


def main():
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    worker = SrADWorker(opt)
    worker.set_gpu_device()
    worker.set_seed()
    worker.set_dataloader()
    worker.set_logging()
    worker.run_train()


if __name__ == "__main__":
    main()
