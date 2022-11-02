from .lsotb import build as build_lsotb


def build_dataset(image_set, args):
    if args.dataset_file == 'lsotb':
        return build_lsotb(image_set, args)
   
    raise ValueError(f'dataset {args.dataset_file} not supported')
