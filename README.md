## Flattened convolutional neural networks

This is a fork from https://raw.githubusercontent.com/jhjin/flattened-cnn, with the extension to support
2D convolution within a channel.

The original package has 1D convolution modules (over channel, in vertical, in horizontal) used in
[Flattened Convolutional Neural Networks for Feedforward Acceleration] (http://arxiv.org/abs/1412.5474)
where we denote the flattened convolution layer as a sequence of one-dimensional filters across all 3D directions.

Our new 2D convolution module PlanarConvolution generalizes the original 1D modules
VerticalConvolution and HorizontalConvolution and can perform 2D convolution within a channel.

The motivation for PlanarConvolution is that when the kernel size is small (e.g. 3x3), it may be more
efficient to have a single PlanarConvolution instead of having a VerticalConvolution and a HorizontalConvolution. 
Saving an extra layer is potentially useful for implementing CNN in thin devices.

To improve the efficiency of LateralConvolution, we added a new module LateralMaskedConvolution that accepts
a mask matrix (of size nInputPlane * nOutputPlane).  All weights are first multiplied by the mask matrix
in an element-wise fashion before applied to the input.  By making the mask matrix highly sparse, one can
improve the efficiency of the resulting layer in a feed-forward neural network.

We also implemented SpatialUpSamplingPeriodic, which in combination with PlanarConvolution can be used to emulate
planar-version of SpatialFullConvolution.

### Install

Choose both or either of `nn`/`cunn` backend packages depending on your computing environment.

```bash
luarocks install https://raw.githubusercontent.com/yin-zhang/flattened-cnn/master/nnconv1d-scm-1.rockspec    # cpu
luarocks install https://raw.githubusercontent.com/yin-zhang/flattened-cnn/master/cunnconv1d-scm-1.rockspec  # cuda
```

or use this command if you already cloned this repo.

```bash
cd nn-conv1d
luarocks make rocks/nnconv1d-scm-1.rockspec
cd ../cunn-conv1d
luarocks make rocks/cunnconv1d-scm-1.rockspec
```


### Available modules

This is a list of available modules.

```lua
nn.LateralConvolution(nInputPlane, nOutputPlane)              -- 1d conv over feature
nn.LateralMaskedConvolution(nInputPlane, nOutputPlane, mask)  -- 1d masked conv over feature
nn.HorizontalConvolution(nInputPlane, nOutputPlane, kL)       -- 1d conv in horizontal
nn.VerticalConvolution(nInputPlane, nOutputPlane, kL)         -- 1d conv in vertical
nn.PlanarConvolution(nInputPlane, nOutputPlane, kW, kH)       -- 2d conv within feature
nn.SpatialUpSamplingPeriodic(scale)                           -- periodic upsampling 
```


### Example

For the original example, run the command below.

```bash
th example.lua
```

For an example on PlanarConvolution, run the command below.

```bash
th example_planar.lua
```

For an example on LateralMaskedConvolution, run the command below.

```bash
th example_masked.lua
```
