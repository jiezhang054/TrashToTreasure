import torch

from config import get_config, output_dim_dict
from data_loader import get_loader
from solver import Solver

def set_seed(seed, device_name):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_default_device(device_name)
    torch.set_default_dtype(torch.float32)


def main():

    config = get_config()
    set_seed(config.seed, config.device_name)

    print("Start loading the data....")


    train_loader = get_loader(config, mode="train", shuffle=True)
    valid_loader = get_loader(config, mode="valid", shuffle=False)
    test_loader = get_loader(config, mode="test", shuffle=False)


    print('Data loading completed!')

    torch.autograd.set_detect_anomaly(True)

    config.n_class = output_dim_dict.get(config.dataset, 1)

    solver = Solver(config,
                    train_loader=train_loader,
                    dev_loader=valid_loader,
                    test_loader=test_loader,
                    is_train=True)
    solver.train_and_eval()


if __name__ == '__main__':
    main()
