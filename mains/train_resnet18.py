import torch

from models.resnet_si_net import ResNetSiNET

from utils.config_parser import parse


def run_experiment():
    config_path = '../configs/train_resnet18.json'
    cfg = parse(config_path)
    device = torch.device("mps")

    model = ResNetSiNET(cfg=cfg.model, num_classes=1000).to(device)
    a = torch.rand((128, 1, 80, 200)).to(device)
    l = torch.randint(0, 1000, (10,)).to(device)
    print(a.shape)
    out = model(a, l)
    print(out.shape)


if __name__ == '__main__':
    run_experiment()
