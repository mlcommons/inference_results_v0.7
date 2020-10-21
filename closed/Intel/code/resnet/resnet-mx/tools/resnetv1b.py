"""ResNetV1bs, implemented in Gluon."""
# pylint: disable=arguments-differ,unused-argument,missing-docstring
from __future__ import division

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm

__all__ = ['ResNetV1b',
           'resnet50_v1b']

class BottleneckV1b(HybridBlock):
    """ResNetV1b BottleneckV1b
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, planes, strides=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None,
                 norm_kwargs=None, last_gamma=False, **kwargs):
        super(BottleneckV1b, self).__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        self.conv1 = nn.Conv2D(channels=planes, kernel_size=1,
                               use_bias=True)
        self.relu1 = nn.Activation('relu')
        self.conv2 = nn.Conv2D(channels=planes, kernel_size=3, strides=strides,
                               padding=dilation, dilation=dilation, use_bias=True)
        self.relu2 = nn.Activation('relu')
        self.conv3 = nn.Conv2D(channels=planes * 4, kernel_size=1, use_bias=True)
        self.relu3 = nn.Activation('relu')
        self.downsample = downsample
        self.dilation = dilation
        self.strides = strides

    def hybrid_forward(self, F, x):
        residual = x

        out = self.conv1(x)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.relu2(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu3(out)

        return out

class ResNetV1b(HybridBlock):
    """ Pre-trained ResNetV1b Model, which produces the strides of 8
    featuremaps at conv5.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    deep_stem : bool, default False
        Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    final_drop : float, default 0.0
        Dropout ratio before the final classification layer.
    use_global_stats : bool, default False
        Whether forcing BatchNorm to use global statistics instead of minibatch statistics;
        optionally set to True if finetuning using ImageNet classification pretrained models.


    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    def __init__(self, block, layers, classes=1000, dilated=False, norm_layer=BatchNorm,
                 norm_kwargs=None, last_gamma=False, deep_stem=False, stem_width=32,
                 avg_down=False, final_drop=0.0, use_global_stats=False,
                 name_prefix='', **kwargs):
        self.inplanes = stem_width*2 if deep_stem else 64
        super(ResNetV1b, self).__init__(prefix=name_prefix)
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if use_global_stats:
            norm_kwargs['use_global_stats'] = True
        self.norm_kwargs = norm_kwargs
        with self.name_scope():
            self.conv1 = nn.Conv2D(channels=64, kernel_size=7, strides=2,
                                    padding=3, use_bias=False)
            self.bn1 = norm_layer(in_channels=64 if not deep_stem else stem_width*2,
                                  **norm_kwargs)
            self.relu = nn.Activation('relu')
            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.layer1 = self._make_layer(1, block, 64, layers[0], avg_down=avg_down,
                                           norm_layer=norm_layer, last_gamma=last_gamma)
            self.layer2 = self._make_layer(2, block, 128, layers[1], strides=2, avg_down=avg_down,
                                           norm_layer=norm_layer, last_gamma=last_gamma)
            self.layer3 = self._make_layer(3, block, 256, layers[2], strides=2,
                                            avg_down=avg_down, norm_layer=norm_layer,
                                            last_gamma=last_gamma)
            self.layer4 = self._make_layer(4, block, 512, layers[3], strides=2,
                                            avg_down=avg_down, norm_layer=norm_layer,
                                            last_gamma=last_gamma)
            self.avgpool = nn.GlobalAvgPool2D()
            self.flat = nn.Flatten()
            self.drop = None
            if final_drop > 0.0:
                self.drop = nn.Dropout(final_drop)
            self.fc = nn.Dense(in_units=512 * block.expansion, units=classes)

    def _make_layer(self, stage_index, block, planes, blocks, strides=1, dilation=1,
                    avg_down=False, norm_layer=None, last_gamma=False):
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
          downsample = nn.HybridSequential(prefix='down%d_'%stage_index)
          with downsample.name_scope():
            downsample.add(nn.Conv2D(channels=planes * block.expansion,
                                    kernel_size=1, strides=strides, use_bias=True))

        layers = nn.HybridSequential(prefix='layers%d_'%stage_index)
        with layers.name_scope():
          layers.add(block(planes, strides, dilation=1,
                            downsample=downsample, previous_dilation=dilation,
                            norm_layer=norm_layer, norm_kwargs=self.norm_kwargs,
                            last_gamma=last_gamma))

          self.inplanes = planes * block.expansion
          for i in range(1, blocks):
              layers.add(block(planes, dilation=dilation,
                                previous_dilation=dilation, norm_layer=norm_layer,
                                norm_kwargs=self.norm_kwargs, last_gamma=last_gamma))

        return layers

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # asymmetric padding before max pooling
        # x = F.pad(x, mode='edge',pad_width=(0,0,0,0,0,1,0,1))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flat(x)
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc(x)

        return x

def resnet50_v1b(**kwargs):
    """Constructs a ResNetV1b-50 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_global_stats : bool, default False
        Whether forcing BatchNorm to use global statistics instead of minibatch statistics;
        optionally set to True if finetuning using ImageNet classification pretrained models.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], name_prefix='resnetv1b_', **kwargs)

    return model