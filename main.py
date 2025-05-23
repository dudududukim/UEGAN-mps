import os
import argparse
from trainer import Trainer
from tester import Tester
from utils import create_folder, setup_seed
from config import get_config
import torch
from munch import Munch
from data_loader import get_train_loader, get_test_loader

# duhyeonkim updated : warning도 무시하지 않겠다
# import warnings
# warnings.simplefilter("error")  # warning to error (better debugging)

def main(args):
    # for fast training. -> mps not avaliable
    # torch.backends.cudnn.benchmark = True

    setup_seed(args.seed)
    
    # create directories if not exist.
    create_folder(args.save_root_dir, args.version, args.model_save_path)
    create_folder(args.save_root_dir, args.version, args.sample_path)
    create_folder(args.save_root_dir, args.version, args.log_path)
    create_folder(args.save_root_dir, args.version, args.val_result_path)
    create_folder(args.save_root_dir, args.version, args.test_result_path)

    if args.mode == 'train':
        # in paper, train(2250(exp) + 2250(raw)), test(500(400(test) + 100(val)))
        loaders = Munch(ref=get_train_loader(root=args.train_img_dir,                   # default='./data/fivek/train'
                                            img_size=args.image_size,                   # default=512
                                            resize_size=args.resize_size,               # default=256
                                            batch_size=args.train_batch_size,           # default=10
                                            shuffle=args.shuffle,                       # default=False
                                            num_workers=args.num_workers,               # 8 https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
                                            drop_last=args.drop_last),                  # sample size // batch size 나머지는 버림
                        val=get_test_loader(root=args.val_img_dir,                      # default='./data/fivek/val'
                                            batch_size=args.val_batch_size,             
                                            shuffle=True,
                                            num_workers=args.num_workers))
        trainer = Trainer(loaders, args)
        trainer.train()
    elif args.mode == 'test':
        loaders = Munch(tes=get_test_loader(root=args.test_img_dir,
                                            img_size=args.test_img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers))
        tester = Tester(loaders, args)
        tester.test()
    else:
        raise NotImplementedError('Mode [{}] is not found'.format(args.mode))

if __name__ == '__main__':

    args = get_config()
    
    # if args.is_print_network:
    #     print(args)
        
    main(args)