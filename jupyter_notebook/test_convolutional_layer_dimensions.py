def output_layer_conv(N, kernel, padding, stride):
    print("Remainder", (N-kernel+2*padding+stride)%(stride)+1 )
    return (N-kernel+2*padding)//(stride)+1

input_size = 256
N=input_size
layers_sizes=[N]
kernel = 2
padding = 0
stride = 2
while (N>5):
    N = output_layer_conv(N,kernel,padding,stride)
    layers_sizes.append(N)
print(layers_sizes)


# Opzione 2
N = dict()
N[0] = 256
N[1] = output_layer_conv(N[0],2,0,1)
N[2] = output_layer_conv(N[1],2,0,1)
N[3] = output_layer_conv(N[2],4,0,2)
N[4] = output_layer_conv(N[3],4,0,2)
N[5] = output_layer_conv(N[4],4,0,2)
N[6] = output_layer_conv(N[5],4,0,2)
N[7] = output_layer_conv(N[6],4,0,2)
N[8] = output_layer_conv(N[7],4,0,2)
N[9] = output_layer_conv(N[8],4,0,2)
N[10]= output_layer_conv(N[9],4,0,2)

print(N)
