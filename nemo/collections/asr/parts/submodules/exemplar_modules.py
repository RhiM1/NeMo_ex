import math
import torch
from torch import nn


def init_weights_kaiming(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, mode='fan_in')


def init_weights_identity(m):
    if type(m) == nn.Linear:
        nn.init.eye_(m.weight)
        # m.weight = nn.Parameter(torch.eye(m.weight.size()[0], m.weight.size()[1]))


class ffnn(nn.Module):
    # Exemplar model incorporating multi-head attention for exemplar weighting, 
    # with separate attention for the acoustic and phonetic information.
    def __init__(
            self,
            inputSize = 40, 
            embedDim = 512,
            outputSize = 39,
            dropout = 0.0
            ):
        super(ffnn, self).__init__()

        self.inputSize = inputSize
        self.outputSize = outputSize
        self.dropout = dropout
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Neural network f
        self.fnn_stack = nn.Sequential(
            nn.Linear(inputSize, embedDim),
            nn.Dropout(p = dropout),
            nn.ReLU(),
            nn.Linear(embedDim, embedDim),
            nn.Dropout(p = dropout),
            nn.ReLU(),
            nn.Linear(embedDim, outputSize)
        )

    def forward(self, input):
        
        return self.fnn_stack(input)



class MHAtt(nn.Module):
    r"""Applies multihead attention.

    Args:
        Q_dim: size of each query sample
        K_dim: size of each key sample (usually the same as Q_dim)
        V_dim: size of each value sample
        dim_per_head: size of the output for each head
        num_heads: number of heads
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``False``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['Q_dim', 'K_dim', 'V_dim', 'dim_per_head', 'num_heads']
    Q_dim: int
    K_dim: int
    V_dim: int
    dim_per_head: int
    V_dim_per_head: int
    num_heads: int
    W_q: torch.Tensor
    W_k: torch.Tensor
    W_v: torch.Tensor

    def __init__(self, Q_dim: int, K_dim: int, V_dim: int,
                dim_per_head: int, V_dim_per_head: int,
                num_heads: int,
                bias: bool = False, device = None, dtype = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MHAtt, self).__init__()
        self.Q_dim = Q_dim
        self.K_dim = K_dim
        self.V_dim = V_dim
        self.dim_per_head = dim_per_head
        self.V_dim_per_head = V_dim_per_head
        self.num_heads = num_heads
        self.bias = bias

        # Weights
        self.W_q = nn.parameter.Parameter(torch.empty((num_heads, Q_dim, dim_per_head), **factory_kwargs))
        self.W_k = nn.parameter.Parameter(torch.empty((num_heads, K_dim, dim_per_head), **factory_kwargs))
        self.W_v = nn.parameter.Parameter(torch.empty((num_heads, V_dim, V_dim_per_head), **factory_kwargs))

        # Final transform
        self.W_0 = nn.Linear(V_dim_per_head * num_heads, V_dim_per_head * num_heads)

        self.sm = nn.Softmax(dim = -1)

        if bias:
            self.bias_q = nn.Parameter(torch.empty((num_heads, dim_per_head), **factory_kwargs), )
            self.bias_k = nn.Parameter(torch.empty((num_heads, dim_per_head), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((num_heads, V_dim_per_head), **factory_kwargs))
        else:
            self.register_parameter('bias_q', None)
            self.register_parameter('bias_k', None)
            self.register_parameter('bias_v', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for head in range(self.num_heads):
            nn.init.kaiming_uniform_(self.W_q[head], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W_k[head], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W_v[head], a=math.sqrt(5))
        if self.bias:
            fan_in_q, _ = nn.init._calculate_fan_in_and_fan_out(self.W_q)
            fan_in_k, _ = nn.init._calculate_fan_in_and_fan_out(self.W_k)
            fan_in_v, _ = nn.init._calculate_fan_in_and_fan_out(self.W_v)
            bound = 1 / math.sqrt(fan_in_q) if fan_in_q > 0 else 0
            nn.init.uniform_(self.bias_q, -bound, bound)
            bound = 1 / math.sqrt(fan_in_k) if fan_in_k > 0 else 0
            nn.init.uniform_(self.bias_k, -bound, bound)
            bound = 1 / math.sqrt(fan_in_v) if fan_in_v > 0 else 0
            nn.init.uniform_(self.bias_v, -bound, bound)

    def forward(self, Q: torch.Tensor, K: torch.tensor, V: torch.tensor, gamma = 1) -> torch.Tensor:

        # batch_size = Q.size()[0]
        # batch_length = Q.size()[1]
        # ex_size = K.size()[0]
        # ex_length = K.size()[1]

        # Q has dimension (miniBatchSize, uttLength1, Q_dim) (Q_dim = featureSize)
        # print("Q.size():", Q.size())
        # print("self.W_q.size():", self.W_q.size())
        # Q_w has dimension (num_heads, miniBatchSize x uttLength1, dim_per_head)
        Q_w = torch.matmul(Q, self.W_q)
        # print("Q_w.size():", Q_w.size())
        # K has dimension (exSize, uttLength2, K_dim) (K_dim = featureSize)
        # print("K.size():", K.size())
        # print("self.W_k.size():", self.W_k.size())
        # K_w has dimension (num_heads, exSize x uttLength2, dim_per_head)
        K_w = torch.matmul(K, self.W_k)
        # print("K_w.size():", K_w.size())
        # V has dimension (exSize x uttLength2, V_dim_per_head)
        # print("V.size():", V.size())
        # print("W_v.size():", self.W_v.size())
        # V_w has dimension (num_heads, exSize x uttLength2, dim_per_head)
        V_w = torch.matmul(V, self.W_v)
        # print("V_w.size():", V_w.size())
        K_w = nn.functional.normalize(K_w, dim = -1)
        # Change K_w to dimension (num_heads, dim_per_head, exSize x uttLength2)
        K_w = torch.transpose(K_w, dim0=1, dim1=2)
        # print("K_w.size():", K_w.size())
        # W_logits has dimension(num_heads, miniBatchSize x uttLength1, exSize x uttLength2)
        W_logits = torch.matmul(Q_w, K_w)
        # print("W_logits.size():", W_logits.size())
        # A has dimension (num_heads, miniBatchSize x uttLength1, V_dim_per_head)
        A = torch.matmul(self.sm(gamma * W_logits), V_w)
        # print("A.size():", A.size())
        # Change A to have dimension (miniBatchSize x uttLength1, num_heads, V_dim_per_head))
        A = torch.transpose(A, dim0 = 0, dim1 = 1)
        # print("A.size():", A.size())
        A = A.reshape(-1, self.num_heads * self.V_dim_per_head)
        # print("A.size():", A.size())
        A = self.W_0(A)
        # print("Final A.size():", A.size())


        return W_logits, A

    def extra_repr(self) -> str:
        return 'Q_dim={}, K_dim={}, V_dim={}, dim_per_head={}, V_dim_per_head={} num_heads={}, bias={}'.format(
            self.Q_dim, self.K_dim, self.V_dim, self.dim_per_head, self.V_dim_per_head, self.num_heads, self.bias is not None
        )



class nnExemplarsMH(nn.Module):
    # Exemplar model incorporating multi-head attention for exemplar weighting, 
    # with separate attention for the acoustic and phonetic information.
    def __init__(
            self, 
            num_phones = 129, 
            inputSize = 40, 
            phoneEmbedSize = 4,
            Q_dim = 32,
            V_dim = 32, 
            numHeads = 2, 
            attDim = 256, 
            dropout_ex_r = 0.0, 
            dropout_ex_g = 0.0
            ):
        super(nnExemplarsMH, self).__init__()

        self.num_phones = num_phones
        self.phoneEmbedSize = phoneEmbedSize
        self.phoneContext = 0
        self.Q_dim = Q_dim
        self.inputSize = inputSize
        self.attDim = attDim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # # Neural network r
        # self.exemplar_stack = ffnn(
        #     inputSize=inputSize,
        #     embedDim=V_dim,
        #     outputSize=V_dim,
        #     dropout=dropout_ex_r
        # )
        # self.exemplar_stack.apply(init_weights_identity)

        # Label embedding for current phone
        self.phoneEmbed_stack = ffnn(
            inputSize=num_phones,
            embedDim=phoneEmbedSize,
            outputSize=phoneEmbedSize,
            dropout=0
        )
        self.phoneEmbed_stack.apply(init_weights_kaiming)

        # # Neural network g      
        # self.encoding_stack = ffnn(
        #     inputSize=inputSize,
        #     embedDim=Q_dim,
        #     outputSize=Q_dim,
        #     dropout=dropout_ex_g
        # )
        # self.encoding_stack.apply(init_weights_identity)

        self.MHAtt = MHAtt(
            Q_dim = Q_dim, 
            K_dim = Q_dim, 
            # V_dim = V_dim + phoneEmbedSize, 
            V_dim = V_dim + phoneEmbedSize, 
            dim_per_head = int(attDim / numHeads), 
            V_dim_per_head = int(attDim / numHeads),
            num_heads = numHeads
            )

    def forward(self, features, ex_features, ex_phones, gamma = 1.0):

        print("submod features size:", features.size())
        print("submod ex_features size:", ex_features.size())
        print("submod ex_phones size:", ex_phones.size())
        batch_size = features.size()[0]
        batch_length = features.size()[1]

        features = features.view(features.size()[0] * features.size()[1], -1)
        ex_features = ex_features.view(ex_features.size()[0] * ex_features.size()[1], -1)

        # Get phone embeddings (learnable)
        # ex_phones has dimension (ex_size)
        # oneHotPhones has dimension (ex_size, num_phones)
        oneHotPhones = torch.nn.functional.one_hot(ex_phones, self.num_phones).float()
        # phoneEmbed has dimensions (ex_size, phoneEmbedSize)
        # print("submod oneHotPhones size:", oneHotPhones.size())
        phoneEmbed = self.phoneEmbed_stack(oneHotPhones.view(oneHotPhones.size()[0] * oneHotPhones.size()[1], -1))
        # print("submod phoneEmbed size:", phoneEmbed.size())

        # # features has dimension (miniBatchSize, uttLength1, featsSize)
        # # ex_features has dimension (exSize, uttLength2, featsSize)
        # # Transform the input features and exemplar features
        # # for MHA for weighting acoustic features
        # # Y_w has dimension (miniBatchSize, uttLength1, Q_dim)
        # Y_w = self.encoding_stack(features)
        # # print("submod Y_w size:", Y_w.size())
        # # D_w has dimension (exSize, uttLength2, Q_dim)
        # D_w = self.encoding_stack(ex_features)
        # # print("submod D_w size:", D_w.size())

        # # Get feature embeddings (learnable)
        # # exFeatEmbed has dimension (exSize, uttLength2, V_dim)
        # exFeatEmbed = self.exemplar_stack(ex_features)
        # # print("submod exFeatEmbed size:", exFeatEmbed.size())

        # Get the weights for each exemplar for each input for the acoustic features
        # W_logits_feats has dimension (num_heads, miniBatchSize, exSize x uttLength2, exSize)
        # A_feats has dimension (miniBatchSize, attDim)
        # ex_phones = torch.unsqueeze(ex_phones, 2)
        W_logits, A_feats = self.MHAtt(
            # Q = Y_w, 
            # K = D_w, 
            # V = torch.cat((exFeatEmbed, phoneEmbed), dim=1),
            Q = features, 
            K = ex_features, 
            V = torch.cat((ex_features, phoneEmbed), dim=1),
            gamma = gamma
        )
        # print("A_feats.size():", A_feats.size())
        A_feats = A_feats.reshape(batch_size, batch_length, self.attDim)
        # print("A_feats.size():", A_feats.size())

        return A_feats, W_logits



class nnExemplarsSimple(nn.Module):
    # Exemplar model incorporating multi-head attention for exemplar weighting, 
    # with separate attention for the acoustic and phonetic information.
    def __init__(
            self, 
            num_phones = 129, 
            inputSize = 176, 
            phoneEmbedSize = 4,
            Q_dim = 32,
            V_dim = 32, 
            numHeads = 2, 
            attDim = 256, 
            dropout_ex_r = 0.0, 
            dropout_ex_g = 0.0
            ):
        super(nnExemplarsSimple, self).__init__()

        # self.num_phones = num_phones
        # self.phoneEmbedSize = phoneEmbedSize
        # self.phoneContext = 0
        # self.Q_dim = Q_dim
        # self.inputSize = inputSize
        # self.attDim = attDim
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.V = nn.Linear(inputSize, inputSize, bias = False)
        self.V.apply(init_weights_identity)


        self.sm = nn.Softmax(dim = 1)


    def forward(self, features, ex_features, ex_phones, gamma = 1.0):

        featuresSize = features.size()
        # ex_features = ex_features[0:int(round(featuresSize[0] / 2, 0))]

        # print("submod features size:", features.size())
        # print("submod ex_features size:", ex_features.size())

        features = features.view(features.size()[0] * features.size()[1], -1)
        ex_features = ex_features.view(ex_features.size()[0] * ex_features.size()[1], -1)

        # print("submod viewed features size:", features.size())
        # print("submod viewed ex_features size:", ex_features.size())

        W = torch.matmul(self.V(features), torch.t(nn.functional.normalize(ex_features, dim = -1)))
        # W = torch.matmul(features, torch.t(nn.functional.normalize(ex_features, dim = -1)))

        # print("submod W size:", W.size())

        A = torch.matmul(self.sm(1 * W), ex_features)

        # print("submod A size:", A.size())

        A = A.reshape(featuresSize)
        
        # print("submod final A size:", A.size())

        return A, W




class nnExemplarsSimplePhones(nn.Module):
    # Exemplar model incorporating multi-head attention for exemplar weighting, 
    # with separate attention for the acoustic and phonetic information.
    def __init__(
            self, 
            num_phones = 129, 
            inputSize = 176, 
            phoneEmbedDim = 4,
            Q_dim = 32,
            V_dim = 32, 
            numHeads = 2, 
            attDim = 256, 
            dropout_ex_r = 0.0, 
            dropout_ex_g = 0.0
            ):
        super(nnExemplarsSimplePhones, self).__init__()

        self.phoneEmbed_stack = nn.Linear(inputSize, inputSize, bias = False)

        self.num_phones = num_phones
        # self.phoneEmbedDim = phoneEmbedDim
        # self.phoneContext = 0
        # self.Q_dim = Q_dim
        # self.inputSize = inputSize
        # self.attDim = attDim
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.phoneEmbed_stack = nn.Linear(num_phones, phoneEmbedDim, bias = False)

        self.V = nn.Linear(inputSize, inputSize, bias = False)
        self.V.apply(init_weights_identity)

        self.correctionLayer =  nn.Linear(inputSize + phoneEmbedDim, inputSize)

        self.sm = nn.Softmax(dim = 1)


    def forward(self, features, ex_features, ex_phones, gamma = 1.0):

        featuresSize = features.size()
        # ex_features = ex_features[0:int(round(featuresSize[0] / 2, 0))]

        # print("submod features size:", features.size())
        # print("submod ex_features size:", ex_features.size())

        features = features.view(features.size()[0] * features.size()[1], -1)
        ex_features = ex_features.view(ex_features.size()[0] * ex_features.size()[1], -1)

        # print("submod ex_phones size:", ex_phones.size())

        oneHotPhones = torch.nn.functional.one_hot(ex_phones, self.num_phones).float()
        # print("submod oneHotPhones size:", oneHotPhones.size())
        phoneEmbed = self.phoneEmbed_stack(oneHotPhones.view(oneHotPhones.size()[0] * oneHotPhones.size()[1], -1))
        # print("submod phoneEmbed size:", phoneEmbed.size())        

        # print("submod viewed features size:", features.size())
        # print("submod viewed ex_features size:", ex_features.size())

        W = torch.matmul(self.V(features), torch.t(nn.functional.normalize(ex_features, dim = -1)))
        # W = torch.matmul(features, torch.t(nn.functional.normalize(ex_features, dim = -1)))

        # print("submod W size:", W.size())

        A = torch.matmul(self.sm(1000 * W), torch.cat((ex_features, phoneEmbed), dim = -1))

        A = self.correctionLayer(A)

        # print("submod A size:", A.size())

        A = A.reshape(featuresSize)
        
        # print("submod final A size:", A.size())

        return A, W