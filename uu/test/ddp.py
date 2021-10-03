import os                                                                                
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist

from uu.utils import shape_infer 
from uu.utils import padding_calc
from uu.layers import maxpool2d, conv2d, sequential, tilesplit, tilecopy
from torch.nn.parameter import Parameter
from uu.utils import correctness_check 

from torch.utils.data import Dataset

class RanTensorDataset(Dataset):
    def __init__(self, chanel,H,W):
        self.samples = []
        for i in range(0,4):
            input = torch.rand(chanel,H,W)
            self.samples.append(input)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]




def main():
    print("hello")
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '10.242.66.108'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.gpus, args=(args,))


Kh = 3
Kw = 3
Ph = 1
Pw = 1
chanel = 1
batch_size = 2

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = conv2d.TiledConv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw),
                                  )   
        self.mxp1 = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        self.conv2d_2 = conv2d.TiledConv2d(in_channels=chanel, 
                                        out_channels=chanel, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        )

        self.mxp2 = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        self.tsplit = tilesplit.TiledSplit()
        self.tcopy = tilecopy.TiledCopy()
        self.block1 = sequential.mSequential(*[self.conv2d_1, self.mxp1, self.conv2d_2, self.mxp2])

    def forward(self, x, H, W, nTh, nTw):
        #nTh, nTw -- num of tiles in H,W
        model_device = next(self.parameters()).device
        N, C, oH, oW, shape_dict = shape_infer.shape_infer_sequence(self.block1, H, W, batch_size, chanel)
        stream_structure = self.block1
        print("N, C, oH, oW", N, C, oH, oW)
        out = torch.zeros(N, C, oH, oW, requires_grad=True).cuda(model_device)
        for i in range(0,nTh): 
            for j in range(0,nTw):
                coord = [i,j]
                print("coord", coord)
                # TODO: here we have to somehow provide static info and num_conv. 
                input_shape = (N,C,H,W)
                output_shape = (N,C,oH,oW)
                info = padding_calc.compute_info_beta([i,j], input_shape, output_shape, nTh, nTw, stream_structure, shape_dict)

                print("++++++++++++++++++++++++++++++++++++++++++++++++")
                input_tile = self.tsplit(x, info, stream_structure[0], model_device, [nTh-1, nTw-1]) # -1 here is to match 0-base
                print("***input tile", input_tile.size())
                out_temp = self.conv2d_1(input_tile, info)
                print("1 out_temp", out_temp[0].size())

                out_temp = self.mxp1(out_temp)
                print("max 1", out_temp[0].size())

                out_temp = self.conv2d_2(out_temp)
                print("2 out_temp", out_temp[0].size())
                out_temp = self.mxp2(out_temp)
                print("max 2", out_temp[0].size())

                
                # use customized copy
                fake_pi = info[0][-11]
                tile_shape = fake_pi.cur_output_shape
                tile_size = [tile_shape[0], tile_shape[1]]
                output_index = fake_pi.input_slice
                print(tile_shape, tile_size, output_index)
                out = self.tcopy(out_temp, out, output_index, tile_size)
                #del info
        return out
      



def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    print ("im rank ", rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    model = Net()
    model.cuda(gpu)

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])


    H = 64
    W = 64
    nTh = 4
    nTw = 4
    batch_size = 1
    # Data loading code
    train_dataset = RanTensorDataset(chanel,H,W)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images ) in enumerate(train_loader):
            images = images.cuda(gpu, non_blocking=True)
            print("img-t", images)
            #print("model", model)
            images.requires_grad = True
            # Forward pass
            print("images size ", images.size())
            outputs = model(images, H, W, nTh, nTw)
            print("output-t", outputs)
            # Backward and optimize
 
            outputs.sum().backward()

            print("img-g",images.grad.size(), images.grad)

            
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))





if __name__ == '__main__':
    main() 