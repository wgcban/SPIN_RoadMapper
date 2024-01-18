from linknet import LinkNet34, LinkNet34MTL
from stack_module import StackHourglassNetMTL
from SPIN import SPIN
from unet import UNet

MODELS = {"LinkNet34MTL": LinkNet34MTL, "StackHourglassNetMTL": StackHourglassNetMTL, "SPIN": SPIN, "UNet": UNet}

MODELS_REFINE = {"LinkNet34": LinkNet34}
