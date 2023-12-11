import segmentation_models_pytorch
from segmentation_models_pytorch.encoders import encoders

# from models.UNetSmall.UNetSmall import UNext_Small_best

DECODERS = [
    "Unet",
    "Linknet",
    "FPN",
    "PSPNet",
    "DeepLabV3",
    "DeepLabV3Plus",
    "PAN",
    "UnetPlusPlus",
]
ENCODERS = list(encoders.keys())

def get_deeplab(encoder, num_classes = 1):
    model = define_model(
        "DeepLabV3Plus",
        encoder,
        num_classes=num_classes,
        encoder_weights="imagenet",
    )
    return model

def get_unet(encoder, num_classes = 1):
    model = define_model(
        "Unet",
        encoder,
        num_classes=num_classes,
        encoder_weights="imagenet",
    )
    return model

def define_model(
    decoder_name,
    encoder_name,
    num_classes=1,
    activation=None,
    encoder_weights="imagenet",
):
    """
    Loads a segmentation architecture.

    Args:
        decoder_name (str): Decoder name.
        encoder_name (str): Encoder name.
        num_classes (int, optional): Number of classes. Defaults to 1.
        pretrained : pretrained original weights
        activation (str or None, optional): Activation of the last layer. Defaults to None.
        encoder_weights (str, optional): Pretrained weights. Defaults to "imagenet".

    Returns:
        torch model: Segmentation model.
    """
    assert decoder_name in DECODERS, "Decoder name not supported"
    assert encoder_name in ENCODERS, "Encoder name not supported"

    decoder = getattr(segmentation_models_pytorch, decoder_name)

    model = decoder(
        encoder_name,
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=activation,
    )
    model.num_classes = num_classes

    return model
