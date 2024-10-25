# %%
"""
# Database
"""

trial = 2

# %%
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
# %%
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import nn, optim

epochs=8000

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

for n_batch  in [4,16,180]:
    string_file = "trial_%03d_batch_%03d"%(trial,n_batch)


    # Define the function as per the problem statement
    def compute_function(x, y, t):
        term1 = (0.2 + 0.1 * np.sin(2 * np.pi * (x - 0.1 * t)) * np.cos(2 * np.pi * y)) * (x > t * y)
        term2 = (0.7 + 0.2 * np.cos(6. * np.pi * x) * np.sin(2 * np.pi * (y - 0.1 * t))) * (x <= t * y)
        return term1 + term2

    # Define the dataset class
    class Function2DDataset(Dataset):
        def __init__(self, t_range=(0, 3), num_samples=1024, grid_size=28):
            """
            t_range: tuple, the range of the parameter t (default: [0, 3])
            num_samples: int, number of samples for different t values
            grid_size: int, number of grid points along x and y axes
            """
            self.t_values = np.linspace(t_range[0], t_range[1], num_samples)
            self.grid_size = grid_size
            self.x_values = np.linspace(0, 1, grid_size)
            self.y_values = np.linspace(0, 1, grid_size)
            self.xx, self.yy = np.meshgrid(self.x_values, self.y_values)

        def __len__(self):
            return len(self.t_values)

        def __getitem__(self, idx):
            t = self.t_values[idx]
            zz = compute_function(self.xx, self.yy, t)
            # Convert to PyTorch tensor
            zz_tensor = torch.tensor(zz, dtype=torch.float32).unsqueeze(0)  # Add channel dimension for CNNs
            t_tensor = torch.tensor(t, dtype=torch.float32)  # Use t as a pseudo-label
            return zz_tensor, t_tensor

    grid_size = 256
    # Create dataset and dataloader
    dataset_train = Function2DDataset(num_samples=180, grid_size=grid_size)  # More samples for training
    dataset_test  = Function2DDataset(num_samples=77, grid_size=grid_size)  # More samples for training
    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)
    test_loader  = DataLoader(dataset_test, batch_size=7, shuffle=True)

    # %%
    import matplotlib.pyplot as plt
    plt.imshow(dataset_train[100][0].squeeze(), cmap='gray')

    # %%
    """
    # Convolutional Autoencoder Denoiser
    """

    # %%
    # Simple CNN model similar to MNIST example
    if trial==1:
        class CNNModel(nn.Module):
            def __init__(self):
                super(CNNModel, self).__init__()
                self.conv1 = nn.Conv2d(  1,   4, 2, stride=1, padding=0) #256->255
                self.conv2 = nn.Conv2d(  4,  16, 2, stride=1, padding=0) #255->254
                self.conv3 = nn.Conv2d( 16,  32, 4, stride=2, padding=0) #254->126
                self.conv4 = nn.Conv2d( 32,  64, 4, stride=2, padding=0) #126->62
                self.conv5 = nn.Conv2d( 64, 128, 4, stride=2, padding=0) # 62->30
                self.conv6 = nn.Conv2d(128, 128, 4, stride=2, padding=0) # 30->14
                self.conv7 = nn.Conv2d(128, 128, 4, stride=2, padding=0) # 14->6
                self.conv8 = nn.Conv2d(128, 128, 4, stride=2, padding=0) #  6->2

                self.batchnorm1 = nn.BatchNorm2d(4)
                self.batchnorm2 = nn.BatchNorm2d(16)
                self.batchnorm3 = nn.BatchNorm2d(32)
                self.batchnorm4 = nn.BatchNorm2d(64)
                self.batchnorm5 = nn.BatchNorm2d(128)
                self.batchnorm6 = nn.BatchNorm2d(128)
                self.batchnorm7 = nn.BatchNorm2d(128)

                self.elu1 = nn.ELU(True)
                self.elu2 = nn.ELU(True)
                self.elu3 = nn.ELU(True)
                self.elu4 = nn.ELU(True)
                self.elu5 = nn.ELU(True)
                self.elu6 = nn.ELU(True)
                self.elu7 = nn.ELU(True)

                ### Flatten layer
                self.flatten = nn.Flatten(start_dim=1)
                ### Linear section
                self.lin_encoder = nn.Linear(512, 30)

                self.lin_decoder = nn.Linear(30, 512)
                self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 2, 2))
                self.convt1 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=0, output_padding=0)
                self.convt2 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=0, output_padding=0)
                self.convt3 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=0, output_padding=0)
                self.convt4 = nn.ConvTranspose2d(128,  64, 4, stride=2, padding=0, output_padding=0)
                self.convt5 = nn.ConvTranspose2d( 64,  32, 4, stride=2, padding=0, output_padding=0)
                self.convt6 = nn.ConvTranspose2d( 32,  16, 4, stride=2, padding=0, output_padding=0)
                self.convt7 = nn.ConvTranspose2d( 16,   4, 2, stride=1, padding=0, output_padding=0)
                self.convt8 = nn.ConvTranspose2d(  4,   1, 2, stride=1, padding=0, output_padding=0)
                self.batchnormt1 = nn.BatchNorm2d(128)
                self.batchnormt2 = nn.BatchNorm2d(128)
                self.batchnormt3 = nn.BatchNorm2d(128)
                self.batchnormt4 = nn.BatchNorm2d(64)
                self.batchnormt5 = nn.BatchNorm2d(32)
                self.batchnormt6 = nn.BatchNorm2d(16)
                self.batchnormt7 = nn.BatchNorm2d(4)
                self.elutr1 = nn.ELU(True)
                self.elutr2 = nn.ELU(True)
                self.elutr3 = nn.ELU(True)
                self.elutr4 = nn.ELU(True)
                self.elutr5 = nn.ELU(True)
                self.elutr6 = nn.ELU(True)
                self.elutr7 = nn.ELU(True)


            def encode(self,x):
                x = self.elu1(self.batchnorm1(self.conv1(x)))
                x = self.elu2(self.batchnorm2(self.conv2(x)))
                x = self.elu3(self.batchnorm3(self.conv3(x)))
                x = self.elu4(self.batchnorm4(self.conv4(x)))
                x = self.elu5(self.batchnorm5(self.conv5(x)))
                x = self.elu6(self.batchnorm6(self.conv6(x)))
                x = self.elu7(self.batchnorm7(self.conv7(x)))
                x = self.conv8(x)
                x = self.lin_encoder(self.flatten(x))
                return x

            def decode(self, x):
                x = self.unflatten(self.lin_decoder(x))
                x = self.elutr1(self.batchnormt1(self.convt1(x)))
                x = self.elutr2(self.batchnormt2(self.convt2(x)))
                x = self.elutr3(self.batchnormt3(self.convt3(x)))
                x = self.elutr4(self.batchnormt4(self.convt4(x)))
                x = self.elutr5(self.batchnormt5(self.convt5(x)))
                x = self.elutr6(self.batchnormt6(self.convt6(x)))
                x = self.elutr7(self.batchnormt7(self.convt7(x)))
                x = self.convt8(x)
                return x

            def forward(self, x):
                x = self.decode(self.encode(x))  # Output the t value
                return x
            

    elif trial==2:
        class CNNModel(nn.Module):
            def __init__(self):
                super(CNNModel, self).__init__()
                self.conv1 = nn.Conv2d(  1,   4, 2, stride=1, padding=0) #256->255
                self.conv2 = nn.Conv2d(  4,  16, 2, stride=1, padding=0) #255->254
                self.conv3 = nn.Conv2d( 16,  32, 4, stride=2, padding=0) #254->126
                self.conv4 = nn.Conv2d( 32,  64, 4, stride=2, padding=0) #126->62
                self.conv5 = nn.Conv2d( 64, 128, 4, stride=2, padding=0) # 62->30
                self.conv6 = nn.Conv2d(128, 128, 4, stride=2, padding=0) # 30->14
                self.conv7 = nn.Conv2d(128, 128, 4, stride=2, padding=0) # 14->6

                self.batchnorm1 = nn.BatchNorm2d(4)
                self.batchnorm2 = nn.BatchNorm2d(16)
                self.batchnorm3 = nn.BatchNorm2d(32)
                self.batchnorm4 = nn.BatchNorm2d(64)
                self.batchnorm5 = nn.BatchNorm2d(128)
                self.batchnorm6 = nn.BatchNorm2d(128)

                self.elu1 = nn.ELU(True)
                self.elu2 = nn.ELU(True)
                self.elu3 = nn.ELU(True)
                self.elu4 = nn.ELU(True)
                self.elu5 = nn.ELU(True)
                self.elu6 = nn.ELU(True)

                ### Flatten layer
                self.flatten = nn.Flatten(start_dim=1)
                ### Linear section
                self.lin_encoder = nn.Linear(4608, 10)

                self.lin_decoder = nn.Linear(10, 4608)
                self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 6, 6))
                self.convt2 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=0, output_padding=0)
                self.convt3 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=0, output_padding=0)
                self.convt4 = nn.ConvTranspose2d(128,  64, 4, stride=2, padding=0, output_padding=0)
                self.convt5 = nn.ConvTranspose2d( 64,  32, 4, stride=2, padding=0, output_padding=0)
                self.convt6 = nn.ConvTranspose2d( 32,  16, 4, stride=2, padding=0, output_padding=0)
                self.convt7 = nn.ConvTranspose2d( 16,   4, 2, stride=1, padding=0, output_padding=0)
                self.convt8 = nn.ConvTranspose2d(  4,   1, 2, stride=1, padding=0, output_padding=0)
                self.batchnormt2 = nn.BatchNorm2d(128)
                self.batchnormt3 = nn.BatchNorm2d(128)
                self.batchnormt4 = nn.BatchNorm2d(64)
                self.batchnormt5 = nn.BatchNorm2d(32)
                self.batchnormt6 = nn.BatchNorm2d(16)
                self.batchnormt7 = nn.BatchNorm2d(4)
                self.elutr2 = nn.ELU(True)
                self.elutr3 = nn.ELU(True)
                self.elutr4 = nn.ELU(True)
                self.elutr5 = nn.ELU(True)
                self.elutr6 = nn.ELU(True)
                self.elutr7 = nn.ELU(True)


            def encode(self,x):
                x = self.elu1(self.batchnorm1(self.conv1(x)))
                x = self.elu2(self.batchnorm2(self.conv2(x)))
                x = self.elu3(self.batchnorm3(self.conv3(x)))
                x = self.elu4(self.batchnorm4(self.conv4(x)))
                x = self.elu5(self.batchnorm5(self.conv5(x)))
                x = self.elu6(self.batchnorm6(self.conv6(x)))
                x = self.conv7(x)
                x = self.lin_encoder(self.flatten(x))
                return x

            def decode(self, x):
                x = self.unflatten(self.lin_decoder(x))
                x = self.elutr2(self.batchnormt2(self.convt2(x)))
                x = self.elutr3(self.batchnormt3(self.convt3(x)))
                x = self.elutr4(self.batchnormt4(self.convt4(x)))
                x = self.elutr5(self.batchnormt5(self.convt5(x)))
                x = self.elutr6(self.batchnormt6(self.convt6(x)))
                x = self.elutr7(self.batchnormt7(self.convt7(x)))
                x = self.convt8(x)
                return x

            def forward(self, x):
                x = self.decode(self.encode(x))  # Output the t value
                return x
        

    #  # Simple CNN model similar to MNIST example
    # class CNNModel(nn.Module):
    #     def __init__(self):
    #         super(CNNModel, self).__init__()
    #         self.conv1 = nn.Conv2d(  1,  24, 2, stride=2, padding=0)  # 256-> 128
    #         self.conv2 = nn.Conv2d( 24,  48, 2, stride=2, padding=0)  # 128-> 64
    #         self.conv3 = nn.Conv2d( 48,  96, 2, stride=2, padding=0)  # 64 -> 32
    #         self.conv4 = nn.Conv2d( 96, 192, 2, stride=2, padding=0)  # 32 -> 16
    #         self.conv5 = nn.Conv2d(192, 378, 2, stride=2, padding=0)  # 16 -> 8
    #         self.conv6 = nn.Conv2d(378, 378, 2, stride=2, padding=0)  # 8  -> 4
    #         self.conv7 = nn.Conv2d(378, 378, 2, stride=2, padding=0)  # 4  -> 2

    #         self.batchnorm1 = nn.BatchNorm2d(24)
    #         self.batchnorm2 = nn.BatchNorm2d(48)
    #         self.batchnorm3 = nn.BatchNorm2d(96)
    #         self.batchnorm4 = nn.BatchNorm2d(192)
    #         self.batchnorm5 = nn.BatchNorm2d(378)
    #         self.batchnorm6 = nn.BatchNorm2d(378)

    #         self.elu1 = nn.ELU(True)
    #         self.elu2 = nn.ELU(True)
    #         self.elu3 = nn.ELU(True)
    #         self.elu4 = nn.ELU(True)
    #         self.elu5 = nn.ELU(True)
    #         self.elu6 = nn.ELU(True)

    #         ### Flatten layer
    #         self.flatten = nn.Flatten(start_dim=1)
    #         ### Linear section
    #         self.lin_encoder = nn.Linear(1512, 10)

    #         self.lin_decoder = nn.Linear(10, 1512)
    #         self.unflatten = nn.Unflatten(dim=1, unflattened_size=(378, 2, 2))
    #         self.convt1 = nn.ConvTranspose2d(378, 378, 2, stride=2, padding=0, output_padding=0)
    #         self.convt2 = nn.ConvTranspose2d(378, 378, 2, stride=2, padding=0, output_padding=0)
    #         self.convt3 = nn.ConvTranspose2d(378, 192, 2, stride=2, padding=0, output_padding=0)
    #         self.convt4 = nn.ConvTranspose2d(192,  96, 2, stride=2, padding=0, output_padding=0)
    #         self.convt5 = nn.ConvTranspose2d( 96,  48, 2, stride=2, padding=0, output_padding=0)
    #         self.convt6 = nn.ConvTranspose2d( 48,  24, 2, stride=2, padding=0, output_padding=0)
    #         self.convt7 = nn.ConvTranspose2d( 24,   1, 2, stride=2, padding=0, output_padding=0)
    #         self.batchnormt1 = nn.BatchNorm2d(378)
    #         self.batchnormt2 = nn.BatchNorm2d(378)
    #         self.batchnormt3 = nn.BatchNorm2d(192)
    #         self.batchnormt4 = nn.BatchNorm2d(96)
    #         self.batchnormt5 = nn.BatchNorm2d(48)
    #         self.batchnormt6 = nn.BatchNorm2d(24)
    #         self.elutr1 = nn.ELU(True)
    #         self.elutr2 = nn.ELU(True)
    #         self.elutr3 = nn.ELU(True)
    #         self.elutr4 = nn.ELU(True)
    #         self.elutr5 = nn.ELU(True)
    #         self.elutr6 = nn.ELU(True)


    #     def encode(self,x):
    #         x = self.elu1(self.batchnorm1(self.conv1(x)))
    #         x = self.elu2(self.batchnorm2(self.conv2(x)))
    #         x = self.elu3(self.batchnorm3(self.conv3(x)))
    #         x = self.elu4(self.batchnorm4(self.conv4(x)))
    #         x = self.elu5(self.batchnorm5(self.conv5(x)))
    #         x = self.elu6(self.batchnorm6(self.conv6(x)))
    #         x = self.conv7(x)
    #         x = self.lin_encoder(self.flatten(x))
    #         return x

    #     def decode(self, x):
    #         x = self.unflatten(self.lin_decoder(x))
    #         x = self.elutr1(self.batchnormt1(self.convt1(x)))
    #         x = self.elutr2(self.batchnormt2(self.convt2(x)))
    #         x = self.elutr3(self.batchnormt3(self.convt3(x)))
    #         x = self.elutr4(self.batchnormt4(self.convt4(x)))
    #         x = self.elutr5(self.batchnormt5(self.convt5(x)))
    #         x = self.elutr6(self.batchnormt6(self.convt6(x)))
    #         x = self.convt7(x)
    #         return x

    #     def forward(self, x):
    #         x = self.decode(self.encode(x))  # Output the t value
    #         return x   


    # %%


    # Initialize model, loss function, and optimizer
    model = CNNModel().to(device)
    criterion = nn.MSELoss()  # Regression task to predict t
    optimizer = optim.Adam(model.parameters(), lr=0.001)



    def train(epoch, model, train_loader, optimizer,  cuda=True):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            data_noise = 0.1*torch.randn(data.shape).to(device)
            data_noise = data # + data_noise

            recon_batch = model(data_noise.to(device))
            loss = criterion(recon_batch.to(device), data.to(device))
            loss.backward()

            train_loss += loss.item() * len(data)
            optimizer.step()

            #if batch_idx % 4 == 0:
            #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6e}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
            #                                                                   100. * batch_idx /len(train_loader),
            #                                                                   loss.item()))

        print('Epoch: {} Average loss: {:.4e}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        return loss.item()

    import time
    tic = time.time()
    toc=time.time()
    losses=np.zeros(epochs)
    for epoch in range(1, epochs + 1):
        losses[epoch-1]=train(epoch, model, train_loader, optimizer, True)
        print("one epoch time ", time.time()-toc)
        toc = time.time()
    toc = time.time()
    print("Total time ", toc-tic)
    print("Average time ", (toc-tic)/epochs)

    # %%
    plt.figure()
    plt.semilogy(losses)
    plt.savefig("loss_convAE_%s.pdf"%string_file)
    plt.close()

    import matplotlib.pyplot as plt
    for batch_idx, (data, labels) in enumerate(test_loader):
        data=data.to(device)
        optimizer.zero_grad()
        recon_batch = model(data)



    plt.figure(figsize=(20, 12))
    for i in range(5):

        print(f"Image {i} with label {labels[i]}")
        plt.subplot(3, 5, 1+i)
        plt.imshow(recon_batch[i, :].view(grid_size, grid_size).detach().cpu().numpy(), cmap='binary')
        plt.colorbar()
        plt.axis('off')
        plt.subplot(3, 5, 6+i)
        plt.imshow(data[i, :, :, :].view(grid_size, grid_size).detach().cpu().numpy(), cmap='binary')
        plt.colorbar()
        plt.axis('off')
        plt.subplot(3, 5, 11+i)
        plt.imshow((recon_batch[i, :, :, :]-data[i, :, :, :]).view(grid_size, grid_size).detach().cpu().numpy(), cmap='binary')
        plt.colorbar()
        plt.axis('off')
    plt.savefig("convAE_trial_%s.pdf"%string_file)

    with open("model_details_%s.txt"%string_file, 'a') as f:
        f.write(str(model))



# %%
#Train denoiser








# # Initialize model, loss function, and optimizer
# denoiser_model = CNNModel().to(device)
# criterion = nn.MSELoss()  # Regression task to predict t
# optimizer = optim.Adam(denoiser_model.parameters(), lr=0.001)


# def train_denoiser(epoch, denoiser_model, old_model, train_loader, optimizer,  cuda=True):
#     denoiser_model.train()
#     train_loss = 0
#     for batch_idx, (data, _) in enumerate(train_loader):
#         data = data.to(device)
#         optimizer.zero_grad()

#         old_model_batch = old_model(data)

#         recon_batch = denoiser_model(old_model_batch.to(device))
#         loss = criterion(recon_batch.to(device), data.to(device))
#         loss.backward()

#         train_loss += loss.item() * len(data)
#         optimizer.step()

# #        if batch_idx % 4 == 0:
# #            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6e}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
# #                                                                           100. * batch_idx /len(train_loader),
# #                                                                           loss.item()))

#     print('====&gt; Epoch: {} Average loss: {:.4f}'.format(
#         epoch, train_loss / len(train_loader.dataset)))
#     return loss.item()

# import time
# tic = time.time()
# toc=time.time()
# losses=np.zeros(epochs)
# for epoch in range(1, epochs + 1):
#     losses[epoch-1]=train_denoiser(epoch, denoiser_model, model, train_loader, optimizer, True)
#     print("one epoch time ", time.time()-toc)
#     toc = time.time()
# toc = time.time()
# print("Total time ", toc-tic)
# print("Average time ", (toc-tic)/epochs)

# # %%
# plt.figure()
# plt.semilogy(losses)
# plt.savefig("loss_convAE_denoiser.pdf")
# plt.close()

# # %%
# import matplotlib.pyplot as plt
# for batch_idx, (data, labels) in enumerate(test_loader):
#     data=data.to(device)
#     optimizer.zero_grad()
#     recon_batch = model(data)
#     denoiser_batch = denoiser_model(recon_batch)


# plt.figure(figsize=(20, 12))
# for i in range(5):

#     print(f"Image {i} with label {labels[i]}")
#     plt.subplot(5, 5, 1+i)
#     plt.imshow(recon_batch[i, :].view(grid_size, grid_size).detach().cpu().numpy(), cmap='binary')
#     plt.colorbar()
#     plt.axis('off')
#     plt.subplot(5, 5, 6+i)
#     plt.imshow(denoiser_batch[i, :, :, :].view(grid_size, grid_size).detach().cpu().numpy(), cmap='binary')
#     plt.colorbar()
#     plt.axis('off')
#     plt.subplot(5, 5, 11+i)
#     plt.imshow(data[i, :, :, :].view(grid_size, grid_size).detach().cpu().numpy(), cmap='binary')
#     plt.colorbar()
#     plt.axis('off')
#     plt.subplot(5, 5, 16+i)
#     plt.imshow((denoiser_batch[i, :, :, :]-data[i, :, :, :]).view(grid_size, grid_size).detach().cpu().numpy(), cmap='binary')
#     plt.colorbar()
#     plt.axis('off')
#     plt.subplot(5, 5, 21+i)
#     plt.imshow((recon_batch[i, :, :, :]-data[i, :, :, :]).view(grid_size, grid_size).detach().cpu().numpy(), cmap='binary')
#     plt.colorbar()
#     plt.axis('off')
# plt.savefig("convAE.pdf")
# plt.close()