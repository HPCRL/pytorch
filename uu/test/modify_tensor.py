import torch

def main():
    H = 200
    W = 200
    input = torch.zeros(1,3,H,W, requires_grad=True)
    print(input)


    r = torch.rand(1,3,5,5)

    input[:,:,0:5,0:5] = r
    print(input)


if __name__=="__main__":
    main()