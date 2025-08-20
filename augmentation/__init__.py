from .diff_mix import *
from .real_mix import *
from .ti_mix import *

AUGMENT_METHODS = {
    "ti-mix": TextualInversionMixup,
    "ti_aug": TextualInversionMixup,
    "real-aug": DDIMLoraMixup,
    "real-mix": DDIMLoraMixup,
    "real-gen": RealGeneration,
    "diff-mix": DDIMLoraMixup,
    "diff-aug": DDIMLoraMixup,
    "diff-gen": DDIMLoraGeneration,
}
