from models.TransUNet.vit_seg_modeling import CONFIGS, VisionTransformer

img_size = 512
patch_size = 16

def get_trans_vit_16():
    config_vit = CONFIGS["ViT-B_16"]
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    net = VisionTransformer(config_vit, img_size=img_size, num_classes=config_vit.n_classes)
    return net

def get_trans_r50_16():
    config_vit = CONFIGS["R50-ViT-B_16"]
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(img_size / patch_size), int(img_size / patch_size))
    net = VisionTransformer(config_vit, img_size=img_size, num_classes=config_vit.n_classes)
    return net
