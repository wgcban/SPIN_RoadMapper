from linknet import LinkNet34, LinkNet34MTL
from stack_module import StackHourglassNetMTL
from SPIN import spin
from unet import unet

MODELS = {"LinkNet34MTL": LinkNet34MTL, "StackHourglassNetMTL": StackHourglassNetMTL, "SPIN": spin, "UNet": unet}

MODELS_REFINE = {"LinkNet34": LinkNet34}
