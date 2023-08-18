import torch


def orthogonality_loss(x, y):
    B, C, W, H = x.size()

    x_T = x.permute(0, 1, 3, 2).contiguous()
    xy = x_T@y
    I = torch.eye(W).cuda(torch.device("cuda:0"))
    I2 = I.unsqueeze(0).unsqueeze(0)
    output = xy - I2
    output = torch.mean(output, dim=1)
    output = torch.norm(output, p=2)

    return output


def similarity_loss(x, y):
    sim = x - y
    output = torch.norm(sim, p='fro', dim=0)
    output = torch.mean(output)

    return output


def regularization(x):
    output = torch.norm(x, p='fro', dim=0)
    output = torch.mean(output)

    return output
