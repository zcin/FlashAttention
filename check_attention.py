import numpy as np
import sys
import torch
import random
import torch.nn.functional as F

ERROR_THRESHOLD = 1e-5

def attention_pytorch(Q, K, V, N, d_model, num_heads):
    d_k = d_model // num_heads
    Q = Q.view(N, num_heads, d_k).transpose(0, 1)  # (num_heads, N, d_k)
    K = K.view(N, num_heads, d_k).transpose(0, 1)
    V = V.view(N, num_heads, d_k).transpose(0, 1)

    output = F.scaled_dot_product_attention(Q, K, V)
    return output.transpose(0, 1).contiguous().view(N, d_model)


def generate_output(N, d_model, num_heads):
    Q = 20 * torch.rand(N, d_model) - 10
    K = 20 * torch.rand(N, d_model) - 10
    V = 20 * torch.rand(N, d_model) - 10

    np.savetxt('Q.txt', Q.cpu().numpy(), fmt='%.12f', delimiter='\n')
    np.savetxt('K.txt', K.cpu().numpy(), fmt='%.12f', delimiter='\n')
    np.savetxt('V.txt', V.cpu().numpy(), fmt='%.12f', delimiter='\n')


def verify_output(filename, N, d_model, num_heads):
    Q = np.loadtxt('Q.txt', dtype=np.float32)
    Q = torch.from_numpy(Q)
    K = np.loadtxt('K.txt', dtype=np.float32)
    K = torch.from_numpy(K)
    V = np.loadtxt('V.txt', dtype=np.float32)
    V = torch.from_numpy(V)
    output = np.loadtxt(filename, dtype=np.float32)
    output = torch.from_numpy(output)

    expected = attention_pytorch(Q, K, V, N, d_model, num_heads).flatten()
    np.savetxt('expected.txt', expected.cpu().numpy(), fmt='%.12f', delimiter='\n')
    assert torch.allclose(expected, output, rtol=1e-5, atol=1e-8), f"{expected=}, {output=}"
    print("Verification passed.")

if __name__ == "__main__":
    if sys.argv[1] == "generate":
        generate_output(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        verify_output(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
