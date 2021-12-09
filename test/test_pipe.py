import numpy as np
from velgen.model import Pipeline, Model
from velgen.layer import FlatLayer, DippingLayer, LinearWaterLayer
from velgen.fold import CosineFold
from velgen.fault import LinearFault
from velgen.salt import GaussianSalt, EllipticSalt

def test_pipe():

    shape = (128,100)
    velseed = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    model = Model(shape, velseed)
    pipe = Pipeline([
        FlatLayer(),
        CosineFold(uniform=False),
        LinearFault(nfaults=2)])
    vel = pipe.generate(model)
    vel.tofile('flatfoldfault2.bin')

    model = Model(shape, velseed)
    pipe = Pipeline([
        FlatLayer(),
        LinearFault(nfaults=3)])
    vel = pipe.generate(model)
    vel.tofile('flatfault3.bin')

    model = Model(shape, velseed)
    pipe = Pipeline([
        DippingLayer(),
        CosineFold(uniform=True),
        LinearFault(nfaults=3)])
    vel = pipe.generate(model)
    vel.tofile('dipfoldfault3.bin')

    model = Model(shape, velseed)
    pipe = Pipeline([
        DippingLayer(),
        CosineFold(uniform=True),
        LinearFault(nfaults=3),
        LinearWaterLayer()])
    vel = pipe.generate(model)
    vel.tofile('dipfoldfault3water.bin')

    model = Model(shape, velseed)
    pipe = Pipeline([
        DippingLayer(),
        LinearFault(nfaults=2)])
    vel = pipe.generate(model)
    vel.tofile('dipfault2.bin')

    model = Model(shape, velseed)
    pipe = Pipeline([
        DippingLayer(),
        LinearFault()])
    vel = pipe.generate(model)
    vel.tofile('dipfaultN.bin')

    model = Model(shape, velseed)
    pipe = Pipeline([
        DippingLayer(),
        CosineFold(uniform=True)])
    vel = pipe.generate(model)
    vel.tofile('dipfold.bin')

    model = Model(shape, velseed)
    pipe = Pipeline([
        DippingLayer()])
    vel = pipe.generate(model)
    vel.tofile('dip.bin')

    model = Model(shape, velseed)
    pipe = Pipeline([
        DippingLayer(),
        CosineFold(uniform=True),
        GaussianSalt()])
    vel = pipe.generate(model)
    vel.tofile('dipfoldsalt.bin')

    model = Model(shape, velseed)
    pipe = Pipeline([
        FlatLayer()])
    vel = pipe.generate(model)
    vel.tofile('flat.bin')

    model = Model(shape, velseed)
    pipe = Pipeline([
        FlatLayer(),
        GaussianSalt()])
    vel = pipe.generate(model)
    vel.tofile('flatsalt.bin')


def test_gom():
    shape = (200,100)
    velseed = [(1.5, 2.0),(2.0,3.5)]
    model = Model(shape, velseed)
    pipe = Pipeline([
        DippingLayer(y_range=(0.15,0.4)),
        GaussianSalt(width_range=(0.1,0.15)),
        LinearWaterLayer(y_range=(0.1,0.2))
        ])
    vel = pipe.generate(model)
    vel.tofile('gom.bin')

def test_gom2():
    shape = (200,100)
    velseed = [1.5,1.8,2.1,2.4,2.7,3.0,3.3,3.6]
    model = Model(shape, velseed)
    pipe = Pipeline([
        DippingLayer(y_range=(0.15,0.9)),
        GaussianSalt(width_range=(0.1,0.15)),
        LinearWaterLayer(y_range=(0.1,0.2))
        ])
    vel = pipe.generate(model)
    vel.tofile('gom2.bin')

def test_gom3():
    shape = (200,100)
    velseed = np.linspace(1.5,3.5,20)
    wmin=0.05
    wmax=0.1
    hmin=0.4
    hmax=0.6

    model = Model(shape, velseed)
    pipe = Pipeline([
        DippingLayer(y_range=(0.1,0.9),minsplit=0.01),
        GaussianSalt(width_range=(wmin,wmax), height_range=(hmin,hmax), penetrate_layer=False),
        LinearWaterLayer(y_range=(0.1,0.2))
        ])
    vel = pipe.generate(model)
    vel.tofile('gom3.bin')

    model = Model(shape, velseed)
    pipe = Pipeline([
        DippingLayer(y_range=(0.1,0.9),minsplit=0.01),
        GaussianSalt(width_range=(wmin,wmax), height_range=(hmin,hmax), penetrate_layer=True),
        LinearWaterLayer(y_range=(0.1,0.2))
        ])
    vel = pipe.generate(model)
    vel.tofile('gom3_penetrate.bin')

def test_ellipse():
    shape = (200,100)
    velseed = np.linspace(1.5,3.5,20)

    model = Model(shape, velseed)
    pipe = Pipeline([
        DippingLayer(y_range=(0.1,0.9),minsplit=0.01),
        EllipticSalt(),
        LinearWaterLayer(y_range=(0.1,0.2))
        ])
    vel = pipe.generate(model)
    vel.tofile('ellipse.bin')

    model = Model(shape, velseed)
    pipe = Pipeline([
        DippingLayer(y_range=(0.1,0.9),minsplit=0.01),
        GaussianSalt(width_range=(0.05,0.1), height_range=(0.4,0.6)),
        EllipticSalt(),
        LinearWaterLayer(y_range=(0.1,0.2))
        ])
    vel = pipe.generate(model)
    vel.tofile('ellipse2.bin')

    model = Model(shape, velseed)
    pipe = Pipeline([
        DippingLayer(y_range=(0.1,0.9),minsplit=0.01),
        EllipticSalt(),
        EllipticSalt(),
        LinearWaterLayer(y_range=(0.1,0.2))
        ])
    vel = pipe.generate(model)
    vel.tofile('ellipse3.bin')
