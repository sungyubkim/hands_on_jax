from typing import Any, Callable, Tuple, Sequence

ModuleType = Any

import jax
import haiku as hk

@dataclass
class Block(hk.Module):
    name : str
    conv : ModuleType
    norm : ModuleType
    act : Callable
    filters : int
    stride : int = 1
    
    def __call__(self, x, train):
        y = x
        y = self.norm(name=f'{self.name}_norm_0')(y, train)
        y = self.act(y)
        y = self.conv(output_channels=self.filters, kernel_shape=3, name=f'{self.name}_conv_0')(y)
        y = self.norm(name=f'{self.name}_norm_1')(y, train)
        y = self.act(y)
        y = self.conv(output_channels=self.filters, kernel_shape=3, stride=self.stride, name=f'{self.name}_conv_1')(y)
        
        if x.shape != y.shape:
            x = self.conv(output_channels=self.filters, kernel_shape=1, stride=self.stride, name=f'{self.name}_shorcut')(x)
        
        return x + y
    
@dataclass
class Bottleneck(hk.Module):
    name : str
    conv : ModuleType
    norm : ModuleType
    act : Callable
    filters : int
    stride : int = 1
    
    def __call__(self, x, train):
        y = x
        y = self.norm(name=f'{self.name}_norm_0')(y, train)
        y = self.act(y)
        y = self.conv(output_channels=self.filters, kernel_shape=1, name=f'{self.name}_conv_0')(y)
        y = self.norm(name=f'{self.name}_norm_1')(y, train)
        y = self.act(y)
        y = self.conv(output_channels=self.filters, kernel_shape=3, stride=self.stride, name=f'{self.name}_conv_1')(y)
        y = self.norm(name=f'{self.name}_norm_2')(y, train)
        y = self.act(y)
        y = self.conv(output_channels=self.filters * 4, kernel_shape=1, name=f'{self.name}_conv_2')(y)
        
        if x.shape != y.shape:
            x = self.conv(output_channels=self.filters * 4, kernel_shape=1, stride=self.stride, name=f'{self.name}_shorcut')(x)
        
        return x + y
    
@dataclass
class ResNet(hk.Module):
    name : str
    block_cls : ModuleType
    stage_sizes : Sequence[int]
    num_filters : Sequence[int]
    strides : Sequence[int]
    num_classes : int
    act : Callable = jax.nn.relu
    
    def __call__(self, x, train=True, print_shape=False):
        conv = partial(
            hk.Conv2D,
            with_bias=False,
        )
        norm = partial(
            hk.BatchNorm,
            create_scale=True,
            create_offset=True,
            decay_rate=0.9,
            eps=1e-5,
        )
        
        if print_shape:
            print(f'input : {x.shape}')
        x = conv(
            output_channels=64,
            kernel_shape=3,
            stride=1,
            name=f'{self.name}_embedding',
        )(x)
        if print_shape:
            print(f'embedding : {x.shape}')
        
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                # only the first block of block_group follows stride
                # other blocks are stride 1.
                stride = (1 if (j > 0) else self.strides[i])
                x = self.block_cls(
                    name=f'block_{i}_{j}',
                    conv=conv,
                    norm=norm,
                    filters=self.num_filters[i],
                    stride=stride,
                    act=self.act
                    )(x, train)
                if print_shape:
                    print(f'block_{i}_{j} : {x.shape}')
        
        x = norm(name=f'{self.name}_final_norm')(x, train)
        x = self.act(x)
        x = x.mean(axis=(1,2))
        if print_shape:
            print(f'representation : {x.shape}')
        x = hk.Linear(self.num_classes, name=f'{self.name}_head')(x)
        if print_shape:
            print(f'classifier head : {x.shape}')
        return x