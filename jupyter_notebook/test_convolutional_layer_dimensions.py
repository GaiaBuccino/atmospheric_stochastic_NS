def output_layer_conv(N, kernel, padding, stride):
    print("Remainder", (N-kernel+padding+stride+1)%(stride) )
    return (N-kernel+padding+stride+1)//(stride)

input_size = 256
N=input_size
layers_sizes=[N]
kernel = 4
padding = 0
stride = 2
while (N>10):
    N = output_layer_conv(N,kernel,padding,stride)
    layers_sizes.append(N)
print(layers_sizes)
