from .swig import build as build_swig

def build_dataset(image_set, args):
    if args.dataset_file == 'swig':
        return build_swig(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
