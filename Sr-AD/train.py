from utils import *
from options import Options

def main():
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    worker = DAEWorker(opt)
    worker.set_gpu_device()
    worker.set_seed()
    worker.set_dataloader()
    worker.set_logging()
    worker.run_train()


if __name__ == "__main__":
    main()
