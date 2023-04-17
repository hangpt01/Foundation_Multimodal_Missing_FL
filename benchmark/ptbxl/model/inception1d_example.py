import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Collection, Union, Any, Iterable
Floats = Union[float, Collection[float]]
ListOrItem = Union[Collection[Any],int,float,str]
OptListOrItem = Optional[ListOrItem]

def listify(p:OptListOrItem=None, q:OptListOrItem=None):
    "Make `p` listy and the same length as `q`."
    if p is None: p=[]
    elif isinstance(p, str):          p = [p]
    elif not isinstance(p, Iterable): p = [p]
    #Rank 0 tensors in PyTorch are Iterable but don't have a length.
    else:
        try: a = len(p)
        except: p = [p]
    n = q if type(q)==int else len(p) if q is None else len(q)
    if len(p)==1: p = p * n
    assert len(p)==n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)

class Flatten(nn.Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"
    def __init__(self, full:bool=False): 
        super(Flatten, self).__init__()
        self.full = full
    def forward(self, x): return x.view(-1) if self.full else x.view(x.size(0), -1)

def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0., actn:Optional[nn.Module]=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers

def cd_adaptiveconcatpool(relevant, irrelevant, module):
    mpr, mpi = module.mp.attrib(relevant,irrelevant)
    apr, api = module.ap.attrib(relevant,irrelevant)
    return torch.cat([mpr, apr], 1), torch.cat([mpi, api], 1)

def attrib_adaptiveconcatpool(self,relevant,irrelevant):
    return cd_adaptiveconcatpool(relevant,irrelevant,self)

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."
    def __init__(self, sz:Optional[int]=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
    def attrib(self,relevant,irrelevant):
        return attrib_adaptiveconcatpool(self,relevant,irrelevant)

def create_head1d(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, bn_final:bool=False, bn:bool=True, act="relu", concat_pooling=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes; added bn and act here"
    lin_ftrs = [2*nf if concat_pooling else nf, nc] if lin_ftrs is None else [2*nf if concat_pooling else nf] + lin_ftrs + [nc] #was [nf, 512,nc]
    ps = listify(ps)
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True) if act=="relu" else nn.ELU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2), Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
        layers += bn_drop_lin(ni,no,bn,p,actn)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)

def conv(in_planes, out_planes, kernel_size=3, stride=1):
    "convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False)

def noop(x): return x

class InceptionBlock1d(nn.Module):
    def __init__(self, ni, nb_filters, kss, stride=1, act='linear', bottleneck_size=32):
        super().__init__()
        self.bottleneck = conv(ni, bottleneck_size, 1, stride) if (bottleneck_size>0) else noop

        self.convs = nn.ModuleList([conv(bottleneck_size if (bottleneck_size>0) else ni, nb_filters, ks) for ks in kss])
        self.conv_bottle = nn.Sequential(nn.MaxPool1d(3, stride, padding=1), conv(ni, nb_filters, 1))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d((len(kss)+1)*nb_filters), nn.ReLU())

    def forward(self, x):
        #print("block in",x.size())
        bottled = self.bottleneck(x)
        out = self.bn_relu(torch.cat([c(bottled) for c in self.convs]+[self.conv_bottle(x)], dim=1))
        return out

class Shortcut1d(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.act_fn=nn.ReLU(True)
        self.conv=conv(ni, nf, 1)
        self.bn=nn.BatchNorm1d(nf)

    def forward(self, inp, out): 
        #print("sk",out.size(), inp.size(), self.conv(inp).size(), self.bn(self.conv(inp)).size)
        #input()
        return self.act_fn(out + self.bn(self.conv(inp)))
        
class InceptionBackbone(nn.Module):
    def __init__(self, input_channels, kss, depth, bottleneck_size, nb_filters, use_residual):
        super().__init__()

        self.depth = depth
        assert((depth % 3) == 0)
        self.use_residual = use_residual

        n_ks = len(kss) + 1
        self.im = nn.ModuleList([InceptionBlock1d(input_channels if d==0 else n_ks*nb_filters,nb_filters=nb_filters,kss=kss, bottleneck_size=bottleneck_size) for d in range(depth)])
        self.sk = nn.ModuleList([Shortcut1d(input_channels if d==0 else n_ks*nb_filters, n_ks*nb_filters) for d in range(depth//3)])    
        
    def forward(self, x):
        
        input_res = x
        for d in range(self.depth):
            print(d, self.im[d])
            x = self.im[d](x)
            if self.use_residual and d % 3 == 2:
                print(d, d // 3, self.sk[d // 3])
                x = (self.sk[d//3])(input_res, x)
                input_res = x.clone()
        return x

class Inception1d(nn.Module):
    '''inception time architecture'''
    def __init__(self,
            num_classes=2,
            input_channels=8,
            kernel_size=40,
            depth=6,
            bottleneck_size=32,
            nb_filters=32,
            use_residual=True,
            lin_ftrs_head=None,
            ps_head=0.5,
            bn_final_head=False,
            bn_head=True,
            act_head="relu",
            concat_pooling=True):
        super().__init__()
        assert(kernel_size>=40)
        kernel_size = [k-1 if k%2==0 else k for k in [kernel_size,kernel_size//2,kernel_size//4]] #was 39,19,9
        
        layers = [InceptionBackbone(
            input_channels=input_channels,
            kss=kernel_size,
            depth=depth,
            bottleneck_size=bottleneck_size,
            nb_filters=nb_filters,
            use_residual=use_residual)
        ]
       
        n_ks = len(kernel_size) + 1
        #head
        head = create_head1d(n_ks*nb_filters, nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        layers.append(head)
        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x)
    
if __name__ == '__main__':
    model = Inception1d(
        num_classes=71,
        input_channels=12,
        use_residual=True,
        ps_head=0.5,
        lin_ftrs_head=[128],
        kernel_size=40,
        depth=6,
        bottleneck_size=32,
        bn_final_head=False,
        bn_head=True,
        act_head="relu",
        concat_pooling=True
    )
    # for parameter in model.parameters():
    #     if isinstance(parameter, nn.Conv1d) or isinstance(parameter, nn.Linear):
    #         nn.init.ones_(parameter.weight)
    #         if hasattr(parameter, 'bias'):
    #             nn.init.ones_(parameter.bias)
    # model.eval()
    # x = torch.ones(128, 12, 250)
    # y = model(x)
    for name, parameter in model.named_parameters():
        # nn.init.ones_(parameter)
        parameter.data.fill_(0.01)
        # else:
        #     print(name)
    model.eval()
    x = torch.ones(128, 12, 250)
    y = model(x)
    print(y)
    print(y.sum())
    x = torch.ones(128, 12, 250)
    y = model(x)
    print(y)
    print(y.sum())
    import pdb; pdb.set_trace()