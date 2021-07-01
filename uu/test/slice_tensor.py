import torch
from typing import List


cpu_input = torch.reshape(torch.arange(0, 4 * 6 , step=1.0, dtype=torch.float), (4 , 6 ))
print(cpu_input)
print(cpu_input.size())
shape_list = list(cpu_input.size())

#suppose tile is [2x3], column major
r = 2
s = 3
for dim0 in range(0, shape_list[0]-r+1):
    for dim1 in range(0, shape_list[1]-s+1):
        print("d1 {} d2 {}".format(dim0, dim1))
        subtensor = cpu_input[dim0:dim0+r, dim1:dim1+s]
        print(subtensor)

print (" === stride 2 ====")

for dim0 in range(0, shape_list[0]-r+1, 2):
    for dim1 in range(0, shape_list[1]-s+1, 2):
        print("d1 {} d2 {}".format(dim0, dim1))
        subtensor = cpu_input[dim0:dim0+r, dim1:dim1+s]
        print(subtensor)
