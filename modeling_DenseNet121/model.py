import torch
import torch.nn as nn

def _bn_function_factory(norm, relu, conv):
  def bn_function(*inputs):
    # TODO Add dimension changes
    concated_features = torch.cat(inputs, 1)
    bottleneck_output = conv(relu(norm(concated_features)))
    return bottleneck_output
  return bn_function

class _DenseLayer(nn.Sequential):
  def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
    super(_DenseLayer, self).__init__()
    self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
    self.add_module('relu1', nn.ReLU(inplace=True)),
    self.add_module('conv1', nn.Conv2d(in_channels=num_input_features,
                                       out_channels=bn_size * growth_rate,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False)),
    self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
    self.add_module('relu2', nn.ReLU(inplace=True)),
    self.add_module('conv2', nn.Conv2d(in_channels=bn_size * growth_rate,
                                       out_channels=growth_rate,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bias=False)),
    self.drop_rate = drop_rate
    self.memory_efficient = memory_efficient
  
  def forward(self, *prev_features):
    # TODO Add dimensions
    bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
    if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_efatures):
      bottleneck_output = cp.checkpoint(bn_function, *prev_features)
    else:
      bottleneck_output = bn_function(*prev_features)
    new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    if self.drop_rate > 0:
      new_features = F.dropout(new_features, p=self.drop_rate,
                               training=self.training)
    return new_features

class _DenseBlock(nn.Module):
  def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
    super(_DenseBlock, self).__init__()
    # num_layers = 6 (i = 0)
    # num_layers = 12 (i = 1)
    # num_layers = 24 (i = 2)
    # num_layers = 16 (i = 3)
    for i in range(num_layers):
      layer = _DenseLayer(
          num_input_features + i * growth_rate,
          growth_rate=growth_rate,
          bn_size=bn_size,
          drop_rate=drop_rate,
          memory_efficient=memory_efficient
      )
      self.add_module('denselayer%d' % (i + 1), layer)
  
  def forward(self, init_features):
    # TODO Add dimension changes
    features = [init_features]
    for name, layer in self.named_children():
      new_features = layer(*features)
      features.append(new_features)
    return torch.cat(features, 1)



class _Transition(nn.Sequential):
  def __init__(self, num_input_features, num_output_features):
    super(_Transition, self).__init__()
    self.add_module('norm', nn.BatchNorm2d(num_input_features))
    self.add_module('relu', nn.ReLU(inplace=True))
    self.add_module('conv', nn.Conv2d(in_channels=num_input_features,
                                      out_channels=num_output_features,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False))
    self.add_module('pool', nn.AvgPool2d(kernel_size=2,
                                         stride=2))

class DenseNet121(nn.Module):
  def __init__(self, 
               growth_rate=32, 
               block_config=(6, 12, 24, 16),
               num_init_featuremaps=64,
               bn_size=4,
               drop_rate=0,
               num_classes=1000,
               memory_efficient=False,
               grayscale=False,
               ):
    super(DenseNet121, self).__init__()

    # First Convolution
    if grayscale:
      in_channels = 1
    else:
      in_channels = 3
    
    self.features = nn.Sequential(OrderedDict([
                                               ('conv0', nn.Conv2d(in_channels=in_channels,
                                                                  out_channels=num_init_featuremaps,
                                                                  kernel_size=7,
                                                                  stride=2,
                                                                  padding=3,
                                                                  bias=False)),
                                               ('norm0', nn.BatchNorm2d(num_features=num_init_featuremaps)),
                                               ('relu0', nn.ReLU(inplace=True)),
                                               ('pool0', nn.MaxPool2d(kernel_size=3,
                                                                      stride=2,
                                                                      padding=1)),
    ]))

    # Each DenseBlock
    num_features = num_init_featuremaps
    
    # num_layers = 0 (6 _DenseBlock)
    # num_layers = 1 (12 _DenseBlock)
    # num_layers = 2 (24 _DenseBlock)
    # num_layers = 3 (16, _DenseBlock)
    for i, num_layers in enumerate(block_config):
      block = _DenseBlock(
          num_layers=num_layers,
          num_input_features=num_features,
          bn_size=bn_size,
          growth_rate=growth_rate,
          drop_rate=drop_rate,
          memory_efficient=memory_efficient
      )
      self.features.add_module('denseblock%d' % (i + 1), block)
      num_features = num_features + num_layers * growth_rate
      if i != len(block_config) - 1:
        trans = _Transition(num_input_features=num_features,
                            num_output_features=num_features // 2)
        self.features.add_module('transition%d' % (i + 1), trans)
        num_features = num_features // 2

    # Final BatchNorm
    self.features.add_module('norm5', nn.BatchNorm2d(num_features))

    # Linear Layer
    self.classifier = nn.Linear(num_features, num_classes)

    # Official init from torch repo
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.constant_(m.bias, 0)
  
  def forward(self, x):
    features = self.features(x)
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    logits = self.classifier(out)
    probas = F.softmax(logits, dim=1)
    return logits, probas
