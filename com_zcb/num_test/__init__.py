import torch
import math
import torch.nn as nn
n = torch.Tensor([ 1.1045e+01, -1.2907e+01, -2.9330e+00, -8.3596e+00, -1.1355e+01,
         -9.6669e+00, -1.2265e+00, -8.8564e+00, -2.5941e-01, -2.3758e+00, 1.1045e+01, -1.2907e+01, -2.9330e+00, -8.3596e+00, -1.1355e+01,
         -9.6669e+00, -1.2265e+00, -8.8564e+00, -2.5941e-01, -2.3758e+00])

# n = n.view(2,-1)
n = n.view(1,10,1,2)
n = torch.squeeze(n)
print(n.size())
# n = n.view(2,5,2)
# n = torch.squeeze(n)
# print(n.size())
# print(n)
# m = nn.Softmax()
#
# print(m(n))

