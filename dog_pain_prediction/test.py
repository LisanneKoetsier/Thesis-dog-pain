from tool.test import test_net
from tool.train import train_net
from tool.test_only_ethogram import test_net_only
from tool.train_only_ethogram import train_net_only
from lib.config_file import cfg
#from lib.data_build import Data_loader
from lib.utils.parser import parse_args, load_config

if __name__ == "__main__":
    
    args = parse_args()
    cfg = load_config(args)
    if cfg.ENABLE_TRAIN and cfg.MODEL.MODEL_NAME == "Ethogram_only":
        train_net_only(cfg)
    elif cfg.ENABLE_TRAIN:
        train_net(cfg)
    if cfg.ENABLE_TEST and cfg.MODEL.MODEL_NAME == "Ethogram_only":
        test_net_only(cfg)
    elif cfg.ENABLE_TEST:
        test_net(cfg)
