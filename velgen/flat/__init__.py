from .flat import LayerDepthGenerator
from .flat import VelocityGenerator


def flat_generator(shape,velseed):
    return VelocityGenerator(shape,velseed)
