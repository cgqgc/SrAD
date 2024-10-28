import os
from torchvision import transforms


def get_transform(opt):
    normalize = transforms.Normalize((0.5,), (0.5,)) if opt.model['in_c'] == 1 else \
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform = transforms.Compose([transforms.ToTensor(),
                                    normalize])
    return transform


def get_data_path(dataset):
    data_root = '/data/qh_20T_share_file/qgc/dataset/MedAD'
    if dataset == 'rsna':
        return os.path.join(data_root, "RSNA")
    elif dataset == 'vin':
        return os.path.join(data_root, "VinCXR")
    elif dataset == 'brain':
        return os.path.join(data_root, "BrainTumor")
    elif dataset == 'lag':
        return os.path.join(data_root, "LAG")
    elif dataset == 'brats':
        return os.path.join(data_root, "BraTS2021")
    elif dataset == 'c16':
        return os.path.join(data_root, "Camelyon16")
    elif dataset == 'oct':
        return os.path.join(os.path.expanduser("~"), "datasets", "OCT2017")
    elif dataset == 'colon':
        return os.path.join(os.path.expanduser("~"), "datasets", "Colon_AD_public")
    elif dataset == 'isic':
        return os.path.join(data_root, "ISIC2018_Task3")
    elif dataset == 'cpchild':
        return os.path.join(data_root, "CP-CHILD/CP-CHILD-A")
    elif dataset == 'pse':
        return os.path.join(data_root, "PSE")
    elif dataset == 'pseseg':
        return os.path.join(data_root, "PSE_SEG")
    elif dataset == 'busi':
        return os.path.join(data_root, "BUSI")
    else:
        raise Exception("Invalid dataset: {}".format(dataset))
