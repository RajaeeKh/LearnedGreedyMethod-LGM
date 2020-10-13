import numpy as np

import scipy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import math
import time

# import scipy.io as sio

import sys
import gc

from pathlib import Path
from skimage import color
from skimage import io

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
import cv2
from scipy import ndimage


from PIL import Image



def normlize_kernel(D_L):
    if(len(D_L.shape)==4):
        return torch.div(D_L, torch.norm(D_L, dim=(2, 3)).view(D_L.shape[0], D_L.shape[1], 1, 1))

    if (len(D_L.shape) == 2):
        return torch.div(D_L, torch.norm(D_L, dim=0,keepdim=True))



def softTHR(x,thr):
    thr2=thr.abs()
    # if(len(x.shape)==4):
    #     thr2 = torch.abs(thr.view(1,-1,1,1))
    return F.relu(x-thr2)-F.relu(-x-thr2)



def calculate_psnr(original_image,reconstructed_one,pixel_range=1):

  # if(len(original_image.shape)==4):
  #   tmp1, tmp2, N_row, N_col = original_image.shape
  # if (len(original_image.shape) == 5):
  #     tmp1, tmp2,tmp3, N_row, N_col = original_image.shape
  assert (len(original_image.shape)==4)
  batch,channels,N_row, N_col = original_image.shape

  MSE = (original_image-reconstructed_one).pow(2).sum(dim=(1,2,3))/(channels*N_row*N_col)
  return 10*torch.log10((pixel_range**2)/(MSE))



def calculate_all_loss_model(model,test_dataset,enable_extra_input=False,enable_ssim=False
                             ,save_location = None):


    if(save_location):
        original_flag = test_dataset.get_name(True)

    loss = 0
    local_sparsity = 0
    data_length = len(test_dataset)
    psnr=0
    if(enable_ssim):
        ssim_model = pytorch_msssim.SSIM(data_range=1, size_average=False, channel=test_dataset[0]['noisy_image'].shape[1]
                                         , nonnegative_ssim=True)
        ssim = 0

    creterion = nn.MSELoss()
    model.set_evaluation_mode(True)
    with torch.no_grad():
        for i in range(len(test_dataset)):
            # print(i)
            sys.stdout.flush()
            sample = test_dataset[i]
            x_data = sample['noisy_image'].clone().detach()
            y_data = sample['image'].clone().detach()
            if(enable_extra_input):
                x_data = {'input':x_data,'output':y_data}
            res = model(x_data)
            if(type(res)is dict):
                denoised_image = res['output']
                if('sparsity' in res.keys()):
                    local_sparsity += res['sparsity'].item()
            else:
                denoised_image = res[1]
            if ('mean' in sample.keys()):
                mean = sample['mean']
                denoised_image+=mean
                y_data += mean

            denoised_image[denoised_image>1]=1.0
            denoised_image[denoised_image<0]=0.0
            loss+=creterion(denoised_image,y_data).item()
            psnr += calculate_psnr(y_data,denoised_image).item()
            if (enable_ssim):
                ssim += ssim_model(y_data,denoised_image).item()
            torch.cuda.empty_cache()
            if(save_location):
                name = sample['name']
                restored_name = save_location + name
                print(restored_name)
                io.imsave(restored_name, denoised_image.squeeze(dim=0).transpose(0, 1).transpose(1, 2)
                          .clone().detach().cpu().numpy())

    if (save_location):
        test_dataset.get_name(original_flag)

    model.set_evaluation_mode(False)

    return loss/data_length , local_sparsity/data_length , ((psnr/data_length,ssim/data_length) if enable_ssim else psnr/data_length)


def train_model(train_dataset,test_dataset, model,model_name,criterion,batch_size=10, epochs=200, learning_rate=0.002,optim_type='Adam',
                load_model = False,test_dataset2=-1,eval_steps=(20,5),enable_scheduler = False,save_load_path=''
                ,enable_extra_input=False,enable_ssim=False,scheduler_step = 20
                ,scheduler_factor = 0.5
                ):
    print('starting training ',model_name)
    assert (eval_steps[0]%eval_steps[1]==0)

    assert ((optim_type=='Adam')or(optim_type=='SGD'))
    print('optimizer=',optim_type)


    if(enable_extra_input):
        print('enable_extra_input mode enabled #$%^&')

    if(optim_type=='Adam'):
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    if(optim_type=='SGD'):
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # test_loss_array = np.zeros((epochs + 1))
    # test_local_sparsity_array = np.zeros((epochs + 1))
    # test_psnr_array = np.zeros((epochs + 1))

    test_loss_array = [0]*(epochs + 1)
    test_local_sparsity_array = [0]*(epochs + 1)
    test_psnr_array = [((0,0) if enable_ssim else 0)]*(epochs + 1)

    max_psnr=0

    # criterion = nn.MSELoss()
    start=0
    if (load_model):
        checkpoint = torch.load(save_load_path+model_name+'_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start = checkpoint['epoch']
        criterion = checkpoint['criterion']

    if(enable_scheduler):
        scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_factor)
        if (load_model):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print('scheduler enabled')

    for epoch in range(start,epochs):
        sys.stdout.flush()
        torch.cuda.empty_cache()
        gc.collect()
        epoch_time = time.time()

        print('epoch = ', epoch)

        if (epoch % eval_steps[1] == 0):

            if (epoch % eval_steps[0] == 0):
                test_loss_array[epoch], test_local_sparsity_array[epoch], test_psnr_array[epoch] = calculate_all_loss_model(
                        model, test_dataset,enable_extra_input,enable_ssim)

                print('testset 1-Test',len(test_dataset),'- epoch ', epoch, 'test_loss = ', test_loss_array[epoch], 'test_local_sparsity = ',
                          test_local_sparsity_array[epoch], 'test_psnr = ', test_psnr_array[epoch])

                sys.stdout.flush()

            if(type(test_dataset2)!=type(-1)):
                torch.cuda.empty_cache()
                gc.collect()

                test_loss, test_local_sparsity, test_psnr = calculate_all_loss_model(
                    model, test_dataset2,enable_extra_input,enable_ssim)

                print('test set 2-Test',len(test_dataset2),'- epoch ', epoch, 'test_loss = ', test_loss, 'test_local_sparsity = ',
                      test_local_sparsity, 'test_psnr = ', test_psnr)

            sys.stdout.flush()


            if((test_psnr_array[epoch][0] if enable_ssim else test_psnr_array[epoch]) > max_psnr):
                max_psnr = (test_psnr_array[epoch][0] if enable_ssim else test_psnr_array[epoch])
                print('updating top model')
                dictt_to_save = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'criterion': criterion
                }
                if (enable_scheduler):
                    dictt_to_save['scheduler_state_dict'] = scheduler.state_dict()
                torch.save(dictt_to_save,save_load_path+ 'top_'+model_name + '_model.pt')


        if (model.has_manually_update_parameters(epoch)):
            assert False
            # with torch.no_grad():
            #     prev_param_num = len(list(model.parameters()))
            #     prev_D_L_w_e = model.D_L_w_e.clone().detach()
            #     model.start_collect_statistics()
            #
            #     test_loss, test_local_sparsity, test_psnr = calculate_all_loss_model(
            #         model, test_dataset)
            #
            #     print(
            #     'test set 1-BSD68- epoch ', epoch, 'test_loss = ', test_loss, 'test_local_sparsity = ',
            #     test_local_sparsity, 'test_psnr = ', test_psnr)
            #
            #     model.update_parameters()
            #     assert(len(list(model.parameters())) == prev_param_num)
            #     print('dist=',torch.dist(prev_D_L_w_e,model.D_L_w_e))
            #
            #     prev_optimizer = optimizer
            #     if (optim_type == 'Adam'):
            #         optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            #
            #     if (optim_type == 'SGD'):
            #         optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            #
            #
            #     optimizer.load_state_dict(prev_optimizer.state_dict())
            #
            #     test_loss, test_local_sparsity, test_psnr = calculate_all_loss_model(
            #         model, test_dataset)
            #
            #     print(
            #         'test set 1-BSD68- epoch ', epoch, 'test_loss = ', test_loss, 'test_local_sparsity = ',
            #         test_local_sparsity, 'test_psnr = ', test_psnr)




        # per = np.random.permutation(range(len(x_train)))
        # split = [batch_size * i for i in range(1, math.ceil(len(x_train) / batch_size))]
        # TODO: may need to optimiza this using parallalisim
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for batch in dataloader:
            channels,N_row, N_col =batch['noisy_image'].shape[2], batch['noisy_image'].shape[3], batch['noisy_image'].shape[4]
            x_data = batch['noisy_image'].view(-1, channels, N_row, N_col)
            y_data = batch['image'].view(-1, channels , N_row, N_col)

            noisy_in = x_data.clone()
            optimizer.zero_grad()
            if (enable_extra_input):
                x_data = {'input': x_data, 'output': y_data}

            res = model(x_data)
            if(type(res) is dict):
                denoised_images = res
                # rain_streak = batch['rain_streak']
                # rain_region = batch['rain_region']
                loss = criterion({'input': noisy_in, 'output': y_data
                                  # ,'rain_streak':rain_streak,'rain_region':rain_region
                                  }, denoised_images)
            else:
                denoised_images = res[1]
                loss = criterion(y_data, denoised_images)

            if(model.has_regularization()):
                loss+=model.get_regularization()

            # #TODO: remove this
            # loss = torch.log10(loss)

            # print(loss.item())
            sys.stdout.flush()
            torch.cuda.empty_cache()
            loss.backward()

            model.mask_params_grads()

            optimizer.step()
            dictt_to_save = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion': criterion
            }
            if (enable_scheduler):
                dictt_to_save['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(dictt_to_save, save_load_path+model_name+'_model.pt')


        if (enable_scheduler):
            scheduler.step()
        epoch_time = time.time()-epoch_time
        print('epoch ', epoch,' time = ',epoch_time)

    test_loss_array[epochs], test_local_sparsity_array[epochs], test_psnr_array[epochs] = calculate_all_loss_model(
                model, test_dataset,enable_extra_input,enable_ssim)

    return (test_loss_array,
            test_local_sparsity_array, test_psnr_array)




class ImageDataSet(Dataset):


    def __init__(self, root_dirs, transform=None,duplicate_number = 1,
                 read_immediatley_from_hardisk = False,getname=False,
                 colored = False,rain_dataset = False,load_extra_info=False):

        assert (type(root_dirs)==type([]))
        self.root_dirs = root_dirs
        self.transform = transform
        self.read_immediatley_from_hardisk = read_immediatley_from_hardisk
        # self.names_list = []
        self.images = []
        self.names = []
        self.getname = getname
        self.colored = colored
        self.rain_dataset = rain_dataset
        self.load_extra_info = load_extra_info
        for root_dir in root_dirs:
            entries = Path(root_dir)
            for entry in entries.iterdir():
                if ((entry.name.split('.')[-1] == 'png') or (entry.name.split('.')[-1] == 'jpg')
                        or(entry.name.split('.')[-1] == 'bmp')):
                    name = entry.name
                    if(rain_dataset):
                        name_list = entry.name.split('.')
                        assert(len(name_list)==2)
                        name_parts = name_list[0].split('-')
                        assert (len(name_parts)==2)
                        if(name_parts[0]!='norain'):
                            continue
                        clean_image_name = root_dir + entry.name
                        noisy_image_name = root_dir + 'rain-'+name_parts[1]+'.'+name_list[1]
                        rain_region = root_dir + 'rainregion-'+name_parts[1]+'.'+name_list[1]
                        rain_streak = root_dir + 'rainstreak-' + name_parts[1] + '.' + name_list[1]

                        name = name_parts[1]+'.'+name_list[1]
                        if (self.read_immediatley_from_hardisk):
                            image = {'image':clean_image_name,'noisy_image':noisy_image_name}
                            if(self.load_extra_info):
                                image['rain_region'] = rain_region
                                image['rain_streak'] = rain_streak
                        else:
                            image = {'image':self.read_image(clean_image_name),'noisy_image':self.read_image(noisy_image_name)}
                            if (self.load_extra_info):
                                image['rain_region'] = self.read_image(rain_region)
                                image['rain_streak'] = self.read_image(rain_streak)

                        image = (int(name_parts[1]),image)
                        name = (int(name_parts[1]),name)


                    else:
                        img_name = root_dir + entry.name
                        if(self.read_immediatley_from_hardisk):
                            image = img_name
                        else:
                            image = self.read_image(img_name)

                    for tmp in range(duplicate_number):
                        self.images.append(image)
                        self.names.append(name)

            if (rain_dataset):
                tuples = sorted(self.images, key=lambda a: a[0])
                self.images = [item[1] for item in tuples]

                tuples = sorted(self.names, key=lambda a: a[0])
                self.names = [item[1] for item in tuples]


    def get_name(self,flag):
        prev_flag = self.getname
        self.getname = flag
        return prev_flag

    def read_image(self,img_name):

        image = io.imread(img_name)
        if (not self.colored):
            image = color.rgb2gray(image)
            image = image.reshape((image.shape[0],image.shape[1],1))


        # if (img_name.split('.')[-1] == 'png'):
        if(np.amax(image)>1):
            image = image.astype('float32') / 255

        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if (self.read_immediatley_from_hardisk):
            if(self.rain_dataset):
                sample = {'image': self.read_image(image['image']), 'noisy_image': self.read_image(image['noisy_image'])}
                if (self.load_extra_info):
                    sample['rain_region'] = self.read_image(image['rain_region'])
                    sample['rain_streak'] = self.read_image(image['rain_streak'])
            else:
                sample = self.read_image(image)
        else:
            sample = image

        if self.transform:
            sample = self.transform(sample)

        if self.getname:
            sample['name']=self.names[idx]

        return sample




class Numpy2d_To_PIL(object):
    def __call__(self, sample):
        return Image.fromarray(sample)

class PIL_To_2dNumpy(object):
    def __call__(self, sample):
        res = np.array(sample)
        assert(len(res.shape)==2)
        return res

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size,colored = False):
        assert isinstance(output_size, (int, tuple))
        self.colored = colored
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):

        if(type(image)==dict):
            shape = image['image'].shape
        else:
            shape = image.shape
        # if(self.colored):
        h, w,tmp = shape
        # else:
        #     h, w = shape

        new_h, new_w = self.output_size
        top = h - new_h
        left = w - new_w
        if(top <=0):
            top = 0
            new_h = h
        else:
            top = np.random.randint(0,top)

        if(left <= 0):
            left = 0
            new_w = w
        else:
            left = np.random.randint(0, w - new_w)

        if (type(image) == dict):
            res = {'image':image['image'][top: top + new_h,
                          left: left + new_w],'noisy_image':image['noisy_image'][top: top + new_h,
                          left: left + new_w]}
            if('rain_streak' in image.keys()):
                res['rain_streak'] = image['rain_streak'][top: top + new_h,
                          left: left + new_w]
            if ('rain_region' in image.keys()):
                res['rain_region'] = image['rain_region'][top: top + new_h,
                                     left: left + new_w]
        else:
            res = image[top: top + new_h,
                          left: left + new_w]

        return res




class AddWhiteGaussianNoise(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, image):
        noise = self.sigma * np.random.randn(*image.shape)
        noisy_image = image+noise
        # noisy_image = image

        return {'image':image,'noisy_image':noisy_image}


class ToTensor(object):
    def __init__(self, device,dtype):
        self.device = device
        self.dtype = dtype

    def __call__(self, sample):
        res = {}
        for i in sample.keys():
            if(len(sample[i].shape)==2):
                # assert False
                res[i] = torch.from_numpy(sample[i]).to(device=self.device,dtype=self.dtype).view(
                    1, 1, sample[i].shape[0], sample[i].shape[1])
            if (len(sample[i].shape) == 3):
                res[i] = torch.from_numpy(sample[i]).to(device=self.device, dtype=self.dtype).transpose(1,2).transpose(0
                    ,1).view(1, sample[i].shape[2], sample[i].shape[0], sample[i].shape[1])

        return res



class Normlize(object):
    def __init__(self, mode='mean',d=15,s1=75,s2=75):
        assert ((mode=='mean')or(mode=='high_pass'))
        self.mode = mode
        self.d=d
        self.s1 = s1
        self.s2 = s2


    # From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
    def get_gaussian_kernel(self,kernel_size=3, sigma=2, channels=1):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).double()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)
        # , padding =int((kernel_size-1)/2),padding_mode = 'gfh' )

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        gaussian_filter

        return gaussian_filter

    def __call__(self, sample):
        if(self.mode=='mean'):
            mean = torch.mean(sample['noisy_image'], dim=(2, 3), keepdim=True)

        if(self.mode=='high_pass'):
            # # kernel_size = 9
            # # sigma = 8
            # kernel_size = 5
            # sigma = 1
            # blur_filter = self.get_gaussian_kernel(kernel_size=kernel_size, sigma=sigma).cuda()
            # pad = int((kernel_size - 1) / 2)
            #
            # blured = blur_filter(F.pad(sample['noisy_image'], (pad, pad, pad, pad), 'reflect'))
            # mean = blured
            mean = cv2.bilateralFilter(sample['noisy_image'], self.d, self.s1, self.s2)

        res = {}
        for i in sample.keys():
            res[i] = sample[i] - mean
        res['mean'] = mean
        return res



class L2_L1_Loss(object):
    def __init__(self):
        self.criterion = nn.MSELoss()
        self.criterion2 = nn.L1Loss()

    def __call__(self, real_images,reconstructed_images):
        return self.criterion(real_images,reconstructed_images) + self.criterion2(real_images,reconstructed_images)


class Log_Loss(object):
    def __call__(self, real_images,reconstructed_images):
        MSE = (real_images - reconstructed_images).pow(2).sum(dim=(1,2,3)) / (real_images.shape[1]*real_images.shape[2]*real_images.shape[3])
        return torch.mean(torch.log10(MSE))


class Log_Loss_2(object):
    def __init__(self):
        self.criterion = nn.MSELoss()
    def __call__(self, real_images,reconstructed_images):
        return (torch.log10(self.criterion(real_images,reconstructed_images)))



class Tmp_Loss(object):
    def __init__(self):
        self.criterion = nn.MSELoss()
    def __call__(self, real_images,reconstructed_images):
        return (torch.log10(
                self.criterion(real_images['output'],reconstructed_images['output'])
                + 0.01*self.criterion(real_images['output'],reconstructed_images['pre_output'])
                + self.criterion(real_images['output'],reconstructed_images['pre_output2'])
                ))

class Pass_Max_Coefficients3(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):

        assert(len(input.shape)==3)
        assert(input.shape[2]==1)

        max_indices = torch.squeeze(torch.argmax(torch.abs(input), dim=1, keepdim=True))
        indices_range = torch.arange(input.shape[0])
        res = torch.zeros_like(input)
        # for i in range(input.shape[0]):
        #     res[i, max_indices[i, 0], indices_range] = torch.mul(input[i, max_indices[i, 0], indices_range],conditions[i,0])

        res[indices_range, max_indices, 0] = input[indices_range, max_indices, 0]

        ctx.save_for_backward(res)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        output, = ctx.saved_tensors
        grad_input = grad_output.clone()
        #TODO
        # grad_input[torch.abs(output) <= 0] = 0
        grad_input[torch.abs(output) <= MP_epsilon] = 0
        return grad_input


class Pass_Max_Coefficients_k(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input,k,abs_flag):

        assert(len(input.shape)==3)
        assert(input.shape[2]==1)

        if(abs_flag):
            coh = torch.abs(input)
        else:
            coh = input.clone()

        # res = torch.zeros_like(input)

        res = torch.zeros(input.shape[0],input.shape[1]*k,input.shape[2],device = input.device,dtype=input.dtype)

        # for tmp in range(k):
        #     max_indices = torch.squeeze(torch.argmax(coh, dim=1, keepdim=True))
        #     indices_range = torch.arange(input.shape[0])
        #     res[indices_range, max_indices, 0] = torch.mul(input[indices_range, max_indices, 0], conditions.to(dtype=input.dtype))
        #     coh[indices_range, max_indices, 0] = 0

        max_vals, max_indices = torch.topk(torch.squeeze(coh,dim=2), k, dim=1,sorted=False)
        # max_vals, max_indices = torch.topk(torch.squeeze(coh, dim=2), k, dim=1, sorted=True)
        indices_range = torch.arange(max_indices.shape[0]).view(-1, 1).expand(input.shape[0], max_indices.shape[1])

        indices2 = (torch.arange(k) * input.shape[1])#.view(1,-1).expand(input.shape[0],k)
        indices2 = indices2.to(device = max_indices.device)
        # res[indices_range,max_indices+indices2,0] = torch.mul(input[indices_range, max_indices, 0], conditions.to(dtype=input.dtype))
        res[indices_range, max_indices + indices2, 0] = input[indices_range, max_indices, 0]


        res = res.view(input.shape[0],k,input.shape[1],1)



        ctx.save_for_backward(res)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        output, = ctx.saved_tensors
        grad_input = grad_output.clone()
        #TODO
        # grad_input[torch.abs(output) <= 0] = 0
        # output2 = torch.sum(output,dim=1)
        grad_input[torch.abs(output) <= MP_epsilon] = 0
        grad_input = torch.sum(grad_input,dim=1)
        return grad_input,None,None



class Rand_Pass_Max_Coefficients(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):

        assert(len(input.shape)==3)
        assert(input.shape[2]==1)


        # max_indices = torch.multinomial(torch.abs(input).squeeze(dim=2),1).squeeze(dim=1)
        tmp = torch.abs(input).squeeze(dim=2)
        tmp[tmp.abs() < 1e-4] = 0
        tmp[tmp.abs() < (tmp.abs().max(dim=1, keepdim=True)[0] * 0.80)] = 0
        max_indices = torch.multinomial(tmp, 1).squeeze(dim=1)
        indices_range = torch.arange(input.shape[0])
        res = torch.zeros_like(input)
        # for i in range(input.shape[0]):
        #     res[i, max_indices[i, 0], indices_range] = torch.mul(input[i, max_indices[i, 0], indices_range],conditions[i,0])

        res[indices_range, max_indices, 0] = input[indices_range, max_indices, 0]

        ctx.save_for_backward(res)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        output, = ctx.saved_tensors
        grad_input = grad_output.clone()
        #TODO
        # grad_input[torch.abs(output) <= 0] = 0
        grad_input[torch.abs(output) <= MP_epsilon] = 0
        return grad_input

class Rand_Pass_Max_Coefficients_k(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input,k,abs_flag):

        assert(len(input.shape)==3)
        assert(input.shape[2]==1)

        if(abs_flag):
            coh = torch.abs(input)
        else:
            coh = input.clone()

        # res = torch.zeros_like(input)

        res = torch.zeros(input.shape[0],input.shape[1]*k,input.shape[2],device = input.device,dtype=input.dtype)

        # for tmp in range(k):
        #     max_indices = torch.squeeze(torch.argmax(coh, dim=1, keepdim=True))
        #     indices_range = torch.arange(input.shape[0])
        #     res[indices_range, max_indices, 0] = torch.mul(input[indices_range, max_indices, 0], conditions.to(dtype=input.dtype))
        #     coh[indices_range, max_indices, 0] = 0
        coh[coh.abs() < 1e-4] = 0
        # coh[coh.abs() < (coh.abs().max(dim=1, keepdim=True)[0] * 0.5)] = 0
        max_indices = torch.multinomial(torch.squeeze(coh,dim=2), k)
        # max_vals, max_indices = torch.topk(torch.squeeze(coh,dim=2), k, dim=1,sorted=False)
        # max_vals, max_indices = torch.topk(torch.squeeze(coh, dim=2), k, dim=1, sorted=True)
        indices_range = torch.arange(max_indices.shape[0]).view(-1, 1).expand(input.shape[0], max_indices.shape[1])

        indices2 = (torch.arange(k) * input.shape[1])#.view(1,-1).expand(input.shape[0],k)
        indices2 = indices2.to(device = max_indices.device)
        # res[indices_range,max_indices+indices2,0] = torch.mul(input[indices_range, max_indices, 0], conditions.to(dtype=input.dtype))
        res[indices_range, max_indices + indices2, 0] = input[indices_range, max_indices, 0]


        res = res.view(input.shape[0],k,input.shape[1],1)



        ctx.save_for_backward(res)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        output, = ctx.saved_tensors
        grad_input = grad_output.clone()
        #TODO
        # grad_input[torch.abs(output) <= 0] = 0
        # output2 = torch.sum(output,dim=1)
        grad_input[torch.abs(output) <= MP_epsilon] = 0
        grad_input = torch.sum(grad_input,dim=1)
        return grad_input,None,None


class Patch_MP_Model(nn.Module):
    # n_row,n_col,m : dictionary dims
    def __init__(self, k, n_row, n_col, m, sigma_noise, early_stopping=False,
                 initial_mode='random', channels=1, dictionary_to_load=None):
        super(Patch_MP_Model, self).__init__()
        assert ((initial_mode == 'random') or (initial_mode == 'DCT') or (initial_mode == 'load'))
        self.k = k
        self.n_row = n_row
        self.n_col = n_col
        self.m = m
        self.sigma_noise = sigma_noise
        self.early_stopping = early_stopping

        self.regularization_flag = False

        initial_dictionary = normlize_kernel(torch.rand(channels * n_row * n_col, m))

        if (initial_mode == 'DCT'):
            # assert(enable_rain_net == False)
            assert (n_row == n_col)
            # assert (m == 4*n_row*n_col)
            D = normlize_kernel(self.Init_DCT(n_row, int(math.sqrt(m))))

            if (self.channels > 1):
                # D = torch.cat(tuple([D for i in range(self.channels)]), dim=0)
                D = normlize_kernel(self.Init_DCT(math.ceil(math.sqrt(channels * n_row * n_col)), int(math.sqrt(m))))
                print(D.shape)
                D = D[:channels * n_row * n_col, :]
            initial_dictionary = normlize_kernel(D)
            self.DCT_to_cat = initial_dictionary.clone()

            print(initial_dictionary.shape)

        if (initial_mode == 'load'):
            initial_dictionary = normlize_kernel(dictionary_to_load.clone())
            assert (initial_dictionary.shape == (n_row * n_col, m))

        self.D_L_w_e = nn.Parameter(initial_dictionary)
        # self.D_L_w_d = nn.Parameter(initial_dictionary)
        self.D_L_d = nn.Parameter(initial_dictionary)

        self.extra_info_flag = False

        if (self.early_stopping):
            stopping_coef = 1.085
            self.register_buffer('stopping_coef', torch.tensor([stopping_coef]))

    def set_evaluation_mode(self, flag):
        pass

    def get_extra_info(self, flag):
        self.extra_info_flag = flag

    def setRegularization(self, flag, mu=1e-5):
        # def setRegularization(self, flag, mu=2e-4):
        self.regularization_flag = flag
        self.regularization_mu = mu

    def has_regularization(self):
        return self.regularization_flag

    def get_regularization(self):

        # total_variation_mu =0.0005

        assert (self.regularization_flag)

        res = self.regularization_mu * (
                self.calcualte_mutual_coherence(self.D_L_w_e) +
                self.calcualte_mutual_coherence(self.D_L_d)
        )

        return res

    def calcualte_mutual_coherence(self, D):
        D_normlized = normlize_kernel(D)
        res = torch.max(torch.abs(
            torch.matmul(D_normlized.t(), D_normlized) - torch.eye(D_normlized.shape[1]).to(device=D_normlized.device,
                                                                                            dtype=D_normlized.dtype)))
        return res

    def forward(self, x):

        D_L_w_e = normlize_kernel(self.D_L_w_e)
        # D_L_w_d = self.D_L_w_d
        D_L_w_d = self.D_L_w_e
        D_norm = D_L_w_d.norm(dim=0).view(1, -1, 1)
        D_L_d = self.D_L_d

        patches = x

        assert (patches.shape[2] == 1)

        denoised_patches = torch.zeros_like(patches)
        remained_patches = torch.arange(patches.shape[0])

        if (self.extra_info_flag):
            denoised_patches_sparsity = torch.zeros(patches.shape[0], device=patches.device, dtype=patches.dtype)

        patches_sparse_coding = torch.zeros(patches.shape[0], D_L_w_e.shape[1], 1, dtype=patches.dtype,
                                            device=patches.device)
        r = patches

        epsilon = 0
        if (self.early_stopping):
            coef = self.stopping_coef
            epsilon = self.sigma_noise * round(math.sqrt(patches.shape[1] * patches.shape[2])) * coef

        for k in range(self.k):
            coherence = torch.matmul(D_L_w_e.t(), r)
            max_func = Pass_Max_Coefficients3.apply
            tmp = max_func(coherence)

            patches_sparse_coding = patches_sparse_coding.clone() + tmp / D_norm  # TODO: division maybe multiply
            r = patches - torch.matmul(D_L_w_d, patches_sparse_coding)  # 1
            true_conditions = (torch.norm(r, dim=1).squeeze(dim=1) > epsilon)

            denoised_patches[remained_patches] = torch.matmul(D_L_d, patches_sparse_coding)
            remained_patches = remained_patches[true_conditions]
            patches = patches[true_conditions]
            patches_sparse_coding = patches_sparse_coding[true_conditions]

            r = patches - torch.matmul(D_L_w_d, patches_sparse_coding)

            if (self.extra_info_flag):
                denoised_patches_sparsity[remained_patches] = k + 1

            if (remained_patches.numel() == 0):
                break

        if (self.extra_info_flag):
            res_data = {}
            res_data['sparsity'] = denoised_patches_sparsity
            res_data['output'] = denoised_patches

            res = res_data
        else:
            res = denoised_patches

        return res


class Rain_Activation():
    def __init__(self):
        self.func = nn.ReLU6()
    def __call__(self, x):
        return self.func(x*6)/6

class Inner_Patch_OMP_Model(nn.Module):
    def __init__(self,k,n_row,n_col,m,sigma_noise,early_stopping=True,
                 initial_mode='DCT',equal_dictionaries = False,
                 add_DC_atom=False,dictionary_multiscale=False,
                 dictionary_to_load=None
                 ,channels=1
                 ,apply_attention_mode = False
                 ,enable_rain_net = False
                 ):
        super(Inner_Patch_OMP_Model, self).__init__()
        assert((initial_mode=='random')or(initial_mode=='DCT')or(initial_mode=='load'))
        # self.k=k
        self.register_buffer('k', torch.tensor([k]))
        self.n_row = n_row
        self.n_col = n_col
        self.m = m
        self.sigma_noise = sigma_noise
        self.early_stopping = early_stopping
        self.equal_dictionaries=equal_dictionaries
        self.add_DC_atom = add_DC_atom
        self.dictionary_multiscale = dictionary_multiscale

        if(dictionary_multiscale):
            assert (n_row >=10)
            assert(n_row==n_col)
            n_row_2 = int(n_row/2)
            m2 = int(m/4)



        self.apply_attention_mode = apply_attention_mode
        if(apply_attention_mode):
            assert (early_stopping == False)
            self.attention_net = Attention_Net(channels*n_row*n_col,k)


        self.channels = channels

        self.rand_omp = False

        self.regularization_flag = False
        self.extra_info_flag = False
        self.evaluation_mode = False
        self.batch_omp_flag = False
        self.extra_input = False
        self.cholesky_decomposition = False


        initial_dictionary = normlize_kernel(torch.rand(channels*n_row * n_col, m))
        if(dictionary_multiscale):
            initial_dictionary2 = normlize_kernel(torch.randn(n_row_2 * n_row_2, m2))

        if(initial_mode=='DCT'):
            # assert(enable_rain_net == False)
            assert (n_row==n_col)
            # assert (m == 4*n_row*n_col)
            D = normlize_kernel(self.Init_DCT(n_row, int(math.sqrt(m))))

            if (self.channels > 1):
                # D = torch.cat(tuple([D for i in range(self.channels)]), dim=0)
                D = normlize_kernel(self.Init_DCT(math.ceil(math.sqrt(channels*n_row*n_col)), int(math.sqrt(m))))
                print(D.shape)
                D = D[:channels*n_row*n_col,:]
            initial_dictionary = normlize_kernel(D)
            self.DCT_to_cat = initial_dictionary.clone()

            if (dictionary_multiscale):
                initial_dictionary2 = normlize_kernel(self.Init_DCT(n_row_2, int(math.sqrt(m2))))



            print(initial_dictionary.shape)

        if (initial_mode == 'load'):
            initial_dictionary = normlize_kernel(dictionary_to_load.clone())
            assert (initial_dictionary.shape ==(n_row*n_col,m) )

        self.enable_rain_net = enable_rain_net
        if(enable_rain_net):
            print('rain mode enabled')
            # self.data_atoms_coeffs = nn.Parameter(0.5*torch.ones(initial_dictionary.shape[1]+1))
            self.data_atoms_coeffs = nn.Parameter(torch.rand(initial_dictionary.shape[1] + 1))
            self.rain_function = Rain_Activation()



        self.D_L_w_e = nn.Parameter(initial_dictionary.clone())
        ones_atom_initial_coef = 2.5
        if (self.add_DC_atom):
            self.ones_D_L_w_e = nn.Parameter(torch.ones(self.channels,dtype = self.D_L_w_e.dtype)*ones_atom_initial_coef)

        if (dictionary_multiscale):
            self.D_L_w_e2 = nn.Parameter(initial_dictionary2.clone())

        self.D_L_d = nn.Parameter(initial_dictionary.clone())
        if (self.add_DC_atom):
            self.ones_D_L_d = nn.Parameter(torch.ones(self.channels, dtype=self.D_L_w_e.dtype) * ones_atom_initial_coef)
        if (dictionary_multiscale):
            self.D_L_d2 = nn.Parameter(initial_dictionary2.clone())

        if (self.early_stopping):
            stopping_coef = 1.085

            if (n_row == 6):
                if ((add_DC_atom, dictionary_multiscale) == (True, True)):
                    # maybe 1.075
                    stopping_coef = 1.1250

                if ((add_DC_atom, dictionary_multiscale) == (False, True)):
                    stopping_coef = 1.1250

                if ((add_DC_atom, dictionary_multiscale) == (True, False)):
                    stopping_coef = 1.1250

                if ((add_DC_atom, dictionary_multiscale) == (False, False)):
                    stopping_coef = 1.1250

            if (n_row == 8):
                if ((add_DC_atom, dictionary_multiscale) == (True, True)):
                    # maybe 1.075
                    stopping_coef = 1.085

                if ((add_DC_atom, dictionary_multiscale) == (False, True)):
                    stopping_coef = 1.085

                if ((add_DC_atom, dictionary_multiscale) == (True, False)):
                    stopping_coef = 1.085

                if ((add_DC_atom, dictionary_multiscale) == (False, False)):
                    stopping_coef = 1.085

            if(n_row==10):
                if((add_DC_atom,dictionary_multiscale)==(True,True)):
                    #maybe 1.075
                    stopping_coef = 1.055

                if ((add_DC_atom, dictionary_multiscale) == (False, True)):
                    stopping_coef = 1.075

                if ((add_DC_atom, dictionary_multiscale) == (True, False)):
                    stopping_coef = 1.060

                if ((add_DC_atom, dictionary_multiscale) == (False, False)):
                    stopping_coef = 1.065

            if (n_row == 12):
                if ((add_DC_atom, dictionary_multiscale) == (True, True)):
                    # maybe 1.075
                    stopping_coef = 1.045

                if ((add_DC_atom, dictionary_multiscale) == (False, True)):
                    stopping_coef = 1.055

                if ((add_DC_atom, dictionary_multiscale) == (True, False)):
                    stopping_coef = 1.045

                if ((add_DC_atom, dictionary_multiscale) == (False, False)):
                    stopping_coef = 1.045

            if (n_row == 14):
                if ((add_DC_atom, dictionary_multiscale) == (True, True)):
                    # maybe 1.075
                    stopping_coef = 1.040

                if ((add_DC_atom, dictionary_multiscale) == (False, True)):
                    stopping_coef = 1.040

                if ((add_DC_atom, dictionary_multiscale) == (True, False)):
                    # stopping_coef = 1.060
                    stopping_coef = 1.040

                if ((add_DC_atom, dictionary_multiscale) == (False, False)):
                    stopping_coef = 1.065

            if (n_row == 16):
                if ((add_DC_atom, dictionary_multiscale) == (True, True)):
                    # maybe 1.075
                    stopping_coef = 1.040

                if ((add_DC_atom, dictionary_multiscale) == (False, True)):
                    stopping_coef = 1.040

                if ((add_DC_atom, dictionary_multiscale) == (True, False)):
                    # stopping_coef = 1.060
                    stopping_coef = 1.020

                if ((add_DC_atom, dictionary_multiscale) == (False, False)):
                    stopping_coef = 1.065




            if (apply_attention_mode):
                stopping_coef = 0.95
                # stopping_coef = 0.90

            self.register_buffer('stopping_coef', torch.tensor([stopping_coef]))
            self.aux_init()



    def set_enable_cholesky_decomposition_flag(self,flag):
        self.cholesky_decomposition = flag

    def show_atoms(self):
        # self.inner_show_atoms(normlize_kernel(self.D_L_w_e),self.channels,'D_L_w_e')
        self.inner_show_atoms(self.D_L_w_e, self.channels, 'D_L_w_e')
        if(self.equal_dictionaries == False):
            D_L_d = self.D_L_d

            if (self.enable_rain_net):
                D_L_d_data = D_L_d.clone()
                D_L_d_data=torch.mul(D_L_d_data,self.rain_function(self.data_atoms_coeffs[1:].view(1,-1)))
                self.inner_show_atoms(D_L_d_data, self.channels, 'D_L_d_data')

                D_L_d_rain = D_L_d.clone()
                D_L_d_rain = torch.mul(D_L_d_rain, 1-self.rain_function(self.data_atoms_coeffs[1:].view(1, -1)))
                self.inner_show_atoms(D_L_d_rain, self.channels, 'D_L_d_rain')
            else:
                self.inner_show_atoms(D_L_d, self.channels, 'D_L_d')




    def inner_show_atoms(self,D,channels,name='D',dpi=300):
        # import matplotlib.pyplot as plt
        assert (channels==1) or (channels==3)
        patches_per_row = round(math.sqrt(D.shape[1]))
        n_row, n_col = self.n_row, self.n_col
        zero_pixels_pad = 1
        rows_num = math.ceil(D.shape[1] / patches_per_row)
        padd_zero_patches_num = (rows_num - (D.shape[1] % rows_num)) % rows_num
        padd_patches = torch.zeros_like(D[:,:padd_zero_patches_num])
        D_new = torch.cat((D,padd_patches),dim=1)

        D_tmp = torch.zeros(1,channels,n_row+2*zero_pixels_pad,n_col+2*zero_pixels_pad)
        D_tmp[:,:,zero_pixels_pad:(n_row+zero_pixels_pad),zero_pixels_pad:(n_col+zero_pixels_pad)] = 1
        unfold = nn.Unfold(kernel_size=(n_row+2*zero_pixels_pad,n_col+2*zero_pixels_pad))
        D_tmp2 = unfold(D_tmp).squeeze(dim=2).squeeze(dim=0)
        indecis = torch.arange(D_tmp2.numel())[D_tmp2>0.1]

        pad_coefficint = D.min()
        # pad_coefficint = 0
        D_new2 = torch.zeros(channels*(n_row+2*zero_pixels_pad)*(n_col+2*zero_pixels_pad),D_new.shape[1]
                             ,dtype=D.dtype,device=D.device)+pad_coefficint
        D_new2[indecis,:] = D_new

        kernel_size = (n_row+2*zero_pixels_pad,n_col+2*zero_pixels_pad)
        fold = nn.Fold(output_size=(rows_num*(n_row+2*zero_pixels_pad),patches_per_row*(n_col+2*zero_pixels_pad)),
                       kernel_size=kernel_size,
                       stride = kernel_size)
        D_image = fold(D_new2.unsqueeze(dim=0)).squeeze(dim=0) + abs(min(0,pad_coefficint))


        if (channels == 3):
            a,b = D_image.min(),D_image.max()
            # D_image = (D_image-a)/(b-a)
            # plt.imshow(D_image.transpose(0,1).transpose(1,2).clone().detach().cpu())
            final_D = D_image.transpose(0,1).transpose(1,2).clone().detach().cpu()
        if (channels == 1):
            # plt.imshow(D_image.squeeze(dim=0).clone().detach().cpu(), cmap='gray')
            final_D = D_image.squeeze(dim=0).clone().detach().cpu()

        # plt.title(name)
        # # axs[3].axis('off')
        # # axs[3].set_xticklabels([])
        # # axs[3].set_yticklabels([])
        # plt.savefig(name + '.png', dpi=dpi)
        io.imsave(name + '.png', final_D.numpy())
        dictt_to_save = {
            'D': D_new.clone().detach().cpu().numpy(),
            'label': 'D'
        }
        # torch.save(dictt_to_save, name+ '.mat')
        from scipy.io import savemat
        savemat(name + '.mat', dictt_to_save)

        return D_new




    def aux_init(self):
        pass

    def set_extra_input_flag(self,flag,type='stopping_thresholds'):
        self.extra_input = flag
        self.extra_input_type = type

    def set_batch_OMP_flag(self,flag):
        self.batch_omp_flag = flag
        if(flag):
            D_L_w_e = self.D_L_w_e

            if (self.dictionary_multiscale):
                tmp_D_L_w_e2 = self.D_L_w_e2
                D_L_w_e2 = self.get_4_shifted_dict(tmp_D_L_w_e2)

            if (self.equal_dictionaries):
                D_L_d = D_L_w_e
                if (self.dictionary_multiscale):
                    D_L_d2 = D_L_w_e2
            else:
                D_L_d = self.D_L_d
                if (self.dictionary_multiscale):
                    D_L_d2 = self.get_4_shifted_dict(self.D_L_d2)

            if (self.normlize_atoms):

                D_L_w_e = normlize_kernel(self.D_L_w_e)

                if (self.dictionary_multiscale):
                    D_L_w_e2 = self.get_4_shifted_dict(normlize_kernel(self.D_L_w_e2))

                if (self.equal_dictionaries):
                    D_L_d = D_L_w_e
                    if (self.dictionary_multiscale):
                        D_L_d2 = D_L_w_e2
                else:
                    D_L_d = self.D_L_d
                    if (self.dictionary_multiscale):
                        D_L_d2 = self.get_4_shifted_dict(self.D_L_d2)

            if (self.dictionary_multiscale):
                D_L_w_e = torch.cat((D_L_w_e, D_L_w_e2), dim=1)
                D_L_d = torch.cat((D_L_d, D_L_d2), dim=1)

            if (self.add_DC_atom):
                D_L_w_e = self.cat_DC_atom(D_L_w_e, self.ones_D_L_w_e.abs())  # TODO: maybe pow2
                D_L_d = self.cat_DC_atom(D_L_d, self.ones_D_L_d.abs())

            if (self.add_DC_atom):
                D_L_w_d = self.cat_DC_atom(self.D_L_w_e, self.ones_D_L_w_e.abs())
            else:
                D_L_w_d = D_L_w_e

            D_L_d = D_L_d.clone()
            D_L_d[:, :(D_L_d.shape[1] - 1)] = torch.div(D_L_d[:, :(D_L_d.shape[1] - 1)],
                                                        D_L_w_d[:, :(D_L_d.shape[1] - 1)].norm(dim=0, keepdim=True))
            self.batch_G = torch.matmul(D_L_w_e.t(), D_L_w_e)
            self.batch_D_L_d = D_L_d
            self.batch_D_L_w_e = D_L_w_e




    # http://www.cs.technion.ac.il/users/wwwb/cgi-bin/tr-get.cgi/2008/CS/CS-2008-08.pdf
    # same notations as the mentioned paper
    def batch_OMP(self, init_alpha, epsilon, G, final_D, target_epsilon, max_sparsity):
        import time
        assert (self.evaluation_mode == True)
        assert (len(init_alpha.shape) == 3)
        tt = time.time()
        epsilon = epsilon.view(-1)
        I = torch.zeros(0, device=G.device, dtype=torch.long)
        # G_I_n = torch.zeros(0,device = G.device,dtype = G.dtype)
        # L_global =torch.zeros(init_alpha.shape[0],max_sparsity+1,max_sparsity+1,device = G.device,dtype = G.dtype)
        prev_delta = torch.zeros(init_alpha.shape[0], device=G.device, dtype=G.dtype)
        # alpha = init_alpha.clone()
        alpha = init_alpha

        restored_signals = torch.zeros(init_alpha.shape[0], final_D.shape[0], 1, device=G.device, dtype=G.dtype)
        remained_patches = torch.arange(init_alpha.shape[0])
        if (self.extra_info_flag):
            restored_patches_sparsity = torch.zeros(init_alpha.shape[0], dtype=torch.int)

        # batch_range = torch.arange(init_alpha.shape[0])
        # tt = time.time()-tt
        # print('stage 0 time =',tt)
        # tt = time.time()
        batch_range = torch.arange(init_alpha.shape[0])
        for k in range(max_sparsity):

            # batch_range = torch.arange(init_alpha.shape[0])
            max_indeceis = torch.argmax(alpha.abs().squeeze(dim=2), dim=1)

            # tt = time.time()-tt
            # print('stage 1 time =',tt)
            # tt = time.time()

            if (k > 0):
                g = G[max_indeceis.unsqueeze(dim=1), I].unsqueeze(dim=2)  # I shape = batch,m TODO: use G_I_n
                # g = G_I_n[batch_range,max_indeceis,:].unsqueeze(dim=2)  # I shape = batch,m TODO: use G_I_n
                w = torch.triangular_solve(g, L, upper=False)[0]
                w_t = w.transpose(1, 2)
                # L_global[batch_range,k,k] = (G[max_indeceis, max_indeceis]-torch.matmul(w_t,w).view(-1)).sqrt()
                # L_global[batch_range,k,:k] = w_t.squeeze(dim=1)
                # L = L_global[:,:(k+1),:(k+1)]
                tmp = torch.cat((L, w_t), dim=1)
                tmpp2 = torch.cat(
                    (torch.zeros_like(w), (G[max_indeceis, max_indeceis].unsqueeze(dim=1).unsqueeze(dim=2)
                                           - torch.matmul(w_t, w)).sqrt()), dim=1)
                L = torch.cat((tmp, tmpp2), dim=2)
                # alpha_0_I_n = torch.cat((alpha_0_I_n,init_alpha[batch_range,max_indeceis,0].view(-1,1,1)),dim=1)

                # tt = time.time()-tt
                # print('stage 2.1 time =',tt)
                # tt = time.time()

            else:
                tmp = G[max_indeceis, max_indeceis].sqrt()
                # L_global[batch_range,0,0] = tmp
                L = tmp.unsqueeze(dim=1).unsqueeze(dim=2)
                # alpha_0_I_n = init_alpha[batch_range,max_indeceis,0].view(-1,1,1)
                # tt = time.time()-tt
                # print('stage 2.2 time =',tt)
                # tt = time.time()

            I = torch.cat((I, max_indeceis.unsqueeze(dim=1)), dim=1)
            alpha_0_I_n = init_alpha[batch_range.unsqueeze(dim=1), I, 0].unsqueeze(dim=2)
            sparse_representation = torch.cholesky_solve(alpha_0_I_n, L)

            # tt = time.time()-tt
            # print('stage 3 time =',tt)
            # tt = time.time()

            # tmp_chech

            # g_tmp = G[max_indeceis,:].unsqueeze(dim=2)

            # tt = time.time()-tt
            # print('stage 4.1 time =',tt)
            # tt = time.time()

            # G_I_n = torch.cat((G_I_n,g_tmp),dim=2)
            G_I_n = G[I, :].transpose(1, 2)

            # tt = time.time()-tt
            # print('stage 4.2 time =',tt)
            # tt = time.time()

            beta = torch.matmul(G_I_n, sparse_representation)

            # tt = time.time()-tt
            # print('stage 4.3 time =',tt)
            # tt = time.time()

            alpha = init_alpha - beta
            beta_I_n = beta[batch_range.unsqueeze(dim=1), I, 0].unsqueeze(dim=2)
            current_delta = torch.matmul(sparse_representation.transpose(1, 2), beta_I_n).view(-1)
            epsilon = epsilon - current_delta + prev_delta
            prev_delta = current_delta

            # tt = time.time()-tt
            # print('stage 5 time =',tt)
            # tt = time.time()

            true_condition = epsilon.sqrt() >= target_epsilon
            false_condition = epsilon.sqrt() < target_epsilon

            batch_final_D = final_D[:, I[false_condition]].transpose(0, 1)
            restored_signals[remained_patches[false_condition]] = torch.matmul(batch_final_D,
                                                                               sparse_representation[
                                                                                   false_condition])

            # tt = time.time()-tt
            # print('stage 6 time =',tt)
            # tt = time.time()

            if (self.extra_info_flag):
                restored_patches_sparsity[true_condition] = (k + 1)

            init_alpha = init_alpha[true_condition]
            epsilon = epsilon[true_condition]
            I = I[true_condition]
            # G_I_n = G_I_n[true_condition]
            L = L[true_condition]
            prev_delta = prev_delta[true_condition]
            alpha = alpha[true_condition]
            remained_patches = remained_patches[true_condition]
            # alpha_0_I_n = alpha_0_I_n[true_condition]
            batch_range = batch_range[:remained_patches.shape[0]]
            # L_global = L_global[true_condition]

            # tt = time.time()-tt
            # print('stage 7 time =',tt)
            # tt = time.time()

            if (remained_patches.numel() == 0):
                break

        if (remained_patches.numel() > 0):
            batch_final_D = final_D[:, I].transpose(0, 1)
            restored_signals[remained_patches] = torch.matmul(batch_final_D, sparse_representation[true_condition])

        if (self.extra_info_flag):
            res_data = {}
            res_data['sparsity'] = restored_patches_sparsity
            res_data['output'] = restored_signals
            restored_signals = res_data

            # tt = time.time()-tt
            # print('stage 8 time =',tt)
            # tt = time.time()

        # tt = time.time()-tt
        # print('stage 9 time =',tt)
        # tt = time.time()

        return restored_signals






    def tune_sigma_coef(self,test_set,calculate_loss_function,overrite_max=False,coeff_range=torch.arange(-0.02,0.02,0.001)):
        original_coef = self.stopping_coef.clone()
        coeff_range = coeff_range + original_coef.item()
        # psnr_array = torch.zeros_like(coeff_range)
        loss_array = torch.zeros_like(coeff_range)
        for i in range(coeff_range.numel()):
            self.stopping_coef[:] = coeff_range[i]
            print(self.stopping_coef)
            psnr=0
            loss=0
            inner_itr = 2
            for j in range(inner_itr):
                res = calculate_loss_function(self,test_set)

                # psnr+=psnr_tmp
                loss+=res[0]
            # psnr /= inner_itr
            loss /= inner_itr
            # psnr_array[i]=psnr
            loss_array[i] = loss


        self.stopping_coef[:] = original_coef
        # max_indes = torch.argmax(psnr_array)
        max_indes = torch.argmin(loss_array)
        max_coef = coeff_range[max_indes]
        print('old coeff = ',original_coef)
        print('new max coeff = ',max_coef)
        if(overrite_max):
            self.stopping_coef[:] = max_coef

        return (coeff_range.detach().numpy(),loss_array.detach().numpy())


    def set_evaluation_mode(self,flag):
        self.evaluation_mode = flag

    def set_rand_OMP(self,flag):
        self.rand_omp = flag



    def get_4_shifted_dict(self,D):
        assert (len(D.shape) == 2)
        n_row = int(math.sqrt(D.shape[0]))
        assert (n_row ** 2 == D.shape[0])
        D_2d = D.t().view(-1, n_row, n_row)
        new_D = torch.zeros(D.shape[1], 4, 2 * n_row, 2 * n_row, dtype=D.dtype, device=D.device)
        for rr in range(4):
            row_shift = rr // 2
            col_shift = rr % 2
            new_D[:, rr, n_row * row_shift:(n_row * row_shift + n_row),
            n_row * col_shift:(n_row * col_shift + n_row)] = D_2d
        new_D = new_D.view(-1, 2 * n_row, 2 * n_row).view(4 * D.shape[1], 4 * D.shape[0]).t()
        return new_D



    def get_extra_info(self,flag):
        self.extra_info_flag = flag

    def setRegularization(self,flag,mu=1e-5):
    # def setRegularization(self, flag, mu=2e-4):
        self.regularization_flag=flag
        self.regularization_mu = mu

    def has_regularization(self):
        return self.regularization_flag

    def get_regularization(self):

        # total_variation_mu =0.0005

        assert(self.regularization_flag)

        res = self.regularization_mu * (
            self.calcualte_mutual_coherence(self.D_L_w_e)+
            (0 if self.equal_dictionaries else self.calcualte_mutual_coherence(self.D_L_d))
        )

        return res

    #taken from: https://github.com/meyerscetbon/Deep-K-SVD/blob/master/Deep_KSVD.py
    def Init_DCT(self,n, m):
        """ Compute the Overcomplete Discrete Cosinus Transform. """
        Dictionary = np.zeros((n, m))
        for k in range(m):
            V = np.cos(np.arange(0, n) * k * np.pi / m)
            if k > 0:
                V = V - np.mean(V)
            Dictionary[:, k] = V / np.linalg.norm(V)
        Dictionary = np.kron(Dictionary, Dictionary)
        Dictionary = Dictionary.dot(np.diag(1 / np.sqrt(np.sum(Dictionary ** 2, axis=0))))
        idx = np.arange(0, n ** 2)
        idx = idx.reshape(n, n, order="F")
        idx = idx.reshape(n ** 2, order="C")
        Dictionary = Dictionary[idx, :]
        Dictionary = torch.from_numpy(Dictionary)
        return Dictionary

    def cat_DC_atom(self,D,dc_const = 2.5):
        assert(len(D.shape)==2)
        if(self.channels>1):
            ones = torch.mul(torch.ones(self.channels, round(D.shape[0]/self.channels), dtype=D.dtype, device=D.device) ,
                             dc_const.view(-1,1)).view(-1,1)
        else:
            ones = torch.ones(D.shape[0],1,dtype=D.dtype,device=D.device)*dc_const
        D_new = torch.cat((D,ones),dim=1)
        return D_new



    def mask_params_grads(self):
        pass



    def calcualte_mutual_coherence(self,D):
        D_normlized = normlize_kernel(D)
        res = torch.max(torch.abs(
            torch.matmul(D_normlized.t(), D_normlized) - torch.eye(D_normlized.shape[1]).to(device=D_normlized.device,dtype=D_normlized.dtype)))
        return res

    def calcualte_coherence_mean(self,D):
        D_normlized = normlize_kernel(D)
        res = torch.mean(torch.abs(
            torch.matmul(D_normlized.t(), D_normlized) - torch.eye(D_normlized.shape[1]).to(device=D_normlized.device,dtype=D_normlized.dtype)))
        return res



    def get_support_ls_2(self,patches,Ds):
        assert (len(Ds.shape) == 3)

        Ds_t = torch.transpose(Ds,1,2)
        Ds_t_Ds = torch.matmul(Ds_t,Ds)

        D_t_b = torch.matmul(Ds_t,patches)


        new_patches_sparse_coding,LU = torch.solve(D_t_b,Ds_t_Ds)


        return new_patches_sparse_coding


    def static_OMP(self,patches,D):
        sparse_coding = torch.zeros(patches.shape[0],D.shape[1],1,device=patches.device,dtype=patches.dtype)
        for i in range(patches.shape[0]):
            b = patches[i]
            r=b
            supp = []
            for k in range(self.k):
                coh = torch.matmul(D.t(),r)
                max_index = torch.argmax(torch.abs(coh))
                supp.append(max_index)
                Ds = D[:,supp]
                x_s = torch.matmul(torch.pinverse(Ds),b)
                r = b - torch.matmul(Ds,x_s)

            sparse_coding[i,supp,0]=x_s[:,0]

        return sparse_coding


    def forward(self, x):

        if (self.evaluation_mode and self.batch_omp_flag):
            patches = x
            G = self.batch_G
            D_L_d = self.batch_D_L_d
            D_L_w_e = self.batch_D_L_w_e
            denoised_patches = self.batch_OMP(torch.matmul(D_L_w_e.t(), patches),
                                              torch.matmul(patches.transpose(1, 2), patches),
                                              G, D_L_d, epsilon, self.k)

            return denoised_patches



        # D_L_w_e = self.D_L_w_e
        #
        # if(self.dictionary_multiscale):
        #     tmp_D_L_w_e2 = self.D_L_w_e2
        #     D_L_w_e2 = self.get_4_shifted_dict(tmp_D_L_w_e2)
        #
        # if(self.equal_dictionaries):
        #     D_L_d = D_L_w_e
        #     if (self.dictionary_multiscale):
        #         D_L_d2 = D_L_w_e2
        # else:
        #     D_L_d = self.D_L_d
        #     if (self.dictionary_multiscale):
        #         D_L_d2 = self.get_4_shifted_dict(self.D_L_d2)



        D_L_w_e = normlize_kernel(self.D_L_w_e)

        if (self.dictionary_multiscale):
            D_L_w_e2 = self.get_4_shifted_dict(normlize_kernel(self.D_L_w_e2))

        if (self.equal_dictionaries):
            D_L_d = D_L_w_e
            if (self.dictionary_multiscale):
                D_L_d2 = D_L_w_e2
        else:
            D_L_d = self.D_L_d
            if (self.dictionary_multiscale):
                D_L_d2 = self.get_4_shifted_dict(self.D_L_d2)



        if (self.dictionary_multiscale):
            D_L_w_e = torch.cat((D_L_w_e,D_L_w_e2),dim=1)
            D_L_d = torch.cat((D_L_d, D_L_d2), dim=1)

        if (self.add_DC_atom):
            D_L_w_e = self.cat_DC_atom(D_L_w_e,self.ones_D_L_w_e.abs())  #TODO: maybe pow2
            D_L_d = self.cat_DC_atom(D_L_d,self.ones_D_L_d.abs())


        D_L_w_d = self.D_L_w_e
        # D_L_w_d = D_L_w_e
        if (self.dictionary_multiscale):
            D_L_w_e2 = self.get_4_shifted_dict(self.D_L_w_e2)
            D_L_w_d = torch.cat((D_L_w_d, D_L_w_e2), dim=1)

        if (self.add_DC_atom):
            D_L_w_d = self.cat_DC_atom(D_L_w_d,self.ones_D_L_w_e.abs())

        patches = x



        if(self.extra_input):
            patches = x['input']
            if(self.extra_input_type=='stopping_thresholds'):
                stopping_thresholds = x['stopping_thresholds']
                assert (stopping_thresholds.shape[0]==patches.shape[0])
                assert (len(stopping_thresholds.shape)==1)
            if(self.extra_input_type=='sparsity'):
                true_cardinality = x['sparsity']
                assert (true_cardinality.shape[0]==patches.shape[0])
                assert (len(true_cardinality.shape)==1)


            x = patches

        in_conditions = (patches.squeeze(dim=2).norm(dim=1) > 1e-5)
        if (in_conditions.all() == False):
            print('in_conditions = ',in_conditions.numel()-in_conditions.sum())
            backup_patches = patches.clone()
            patches = patches[in_conditions]

        original_patches = patches.clone()
        # patches_shape = patches.shape


        denoised_patches = torch.zeros_like(patches)
        remained_patches = torch.arange(patches.shape[0])

        if(self.enable_rain_net):
            data_coefs = self.rain_function(self.data_atoms_coeffs)
            patches_data_atoms = torch.zeros(0, dtype=self.data_atoms_coeffs.dtype, device=self.data_atoms_coeffs.device)
            rain_component_patches = torch.zeros_like(denoised_patches)



        Ds = torch.zeros(0,dtype=x.dtype,device=x.device)
        Ds2 = torch.zeros_like(Ds)


        if(self.extra_info_flag):
            denoised_patches_sparsity = torch.zeros(patches.shape[0],device = patches.device,dtype=patches.dtype)
            support_probability = torch.ones(patches.shape[0],device = patches.device,dtype=patches.dtype)
            chosen_support = torch.zeros(patches.shape[0],self.k,device = patches.device,dtype=torch.int64)+(-1)


        r = patches

        epsilon = 0
        # print(self.early_stopping)
        # print(self.extra_input)
        if(self.early_stopping):
            # print('early_stopping')
            coef = self.stopping_coef
            epsilon = self.sigma_noise*round(math.sqrt(patches.shape[1]* patches.shape[2]))*coef
            if (self.extra_input and (self.extra_input_type != 'rain_weights')):
                if (self.extra_input_type == 'stopping_thresholds'):
                    # print('exact stopping threshold')
                    epsilon = stopping_thresholds
                else:
                    epsilon = 0


        # print('eps=', epsilon)


        if(self.apply_attention_mode):
            tmp_res = torch.zeros(r.shape[0],r.shape[1],self.k.item(),dtype = r.dtype,device=r.device)


        coherence = torch.matmul(D_L_w_e.t(), r)



        coherence_conditions = coherence.abs().squeeze(dim=2).max(dim=1)[0] > 1e-4
        if(coherence_conditions.all()==False):
            # import matplotlib.pyplot as plt
            bad_patches = patches[coherence_conditions==False]
            print('bad patches num = ', bad_patches.shape[0])
            if(self.channels > 1):
                R,G,B = bad_patches[:,:n_row*n_col,0],bad_patches[:,n_row*n_col:2*n_row*n_col,0],bad_patches[:,2*n_row*n_col:,0]
                R, G, B = R.view(-1,n_row,n_col,1),G.view(-1,n_row,n_col,1),B.view(-1,n_row,n_col,1)
                bad_patches = torch.cat((R,G,B),dim=3)
                dictt_to_save = {
                    'bad_patches': bad_patches
                }
                torch.save(dictt_to_save,'bad_patches.pt')
                # plt.imshow((image + mean).squeeze(dim=0).transpose(0, 1).transpose(1, 2).cpu())
                # plt.show()
                # true_true_patches = patches
                # true_indecis = (coherence_conditions)

        for k in range(self.k):
            if(self.rand_omp):
                max_func = Rand_Pass_Max_Coefficients.apply
            else:
                max_func = Pass_Max_Coefficients3.apply

            if(self.evaluation_mode and (self.rand_omp == False)):
                max_vals,max_indices = torch.max(coherence.abs().squeeze(dim=2), dim=1)
                new_patch_atoms = D_L_w_d[:,max_indices].t().unsqueeze(dim=2)
                new_patch_atoms_2 = D_L_d[:,max_indices].t().unsqueeze(dim=2)
                if (self.extra_info_flag):
                    support_probability[remained_patches] = support_probability[remained_patches].clone()*(
                        max_vals/(coherence.abs()).squeeze(dim=2).sum(dim=1))
                    chosen_support[remained_patches,k] = max_indices
                if (self.enable_rain_net):
                    current_data_atoms = data_coefs[max_indices].view(-1,1,1)
                    patches_data_atoms = torch.cat((patches_data_atoms,current_data_atoms),dim=1)

            else:
                tmp = max_func(coherence)
                tmp2 = torch.abs(tmp)
                max_vals,max_indices = torch.max(tmp2, dim=1, keepdim=True)
                tmp2 = torch.div(tmp2,max_vals)
                new_patch_atoms = torch.matmul(D_L_w_d,tmp2)
                new_patch_atoms_2 = torch.matmul(D_L_d,tmp2)
                if (self.extra_info_flag):
                    support_probability[remained_patches] = support_probability[remained_patches].clone()*(
                        max_vals.squeeze(dim=2).squeeze(dim=1)/(coherence.abs()).squeeze(dim=2).sum(dim=1))
                    chosen_support[remained_patches,k] = max_indices.squeeze(dim=2).squeeze(dim=1)

                if (self.enable_rain_net):
                    current_data_atoms = data_coefs[max_indices.squeeze(dim=2).squeeze(dim=1)].view(-1,1,1)
                    patches_data_atoms = torch.cat((patches_data_atoms,current_data_atoms),dim=1)


            if(self.cholesky_decomposition):
                if(k==0):
                    L = torch.matmul(new_patch_atoms.transpose(1,2),new_patch_atoms).sqrt()
                    correlation_array = torch.matmul(new_patch_atoms.transpose(1, 2), patches)
                else:
                    correlation_array = torch.cat(
                        (correlation_array, torch.matmul(new_patch_atoms.transpose(1, 2), patches)), dim=1)
                    w = torch.triangular_solve(torch.matmul(Ds.transpose(1,2),new_patch_atoms), L, upper=False)[0]
                    col1 = torch.cat((L,w.transpose(1,2)),dim=1)
                    col2 = torch.cat((torch.zeros_like(w),(torch.matmul(new_patch_atoms.transpose(1,2),new_patch_atoms)-
                                                           torch.matmul(w.transpose(1,2),w)).sqrt()),dim=1)
                    L = torch.cat((col1,col2),dim=2)

                patches_sparse_coding = torch.cholesky_solve(correlation_array, L)



            Ds = torch.cat((Ds, new_patch_atoms), dim=2)
            Ds2 = torch.cat((Ds2, new_patch_atoms_2), dim=2)

            if ( not self.cholesky_decomposition):
                patches_sparse_coding = self.get_support_ls_2(patches,Ds)


            r = patches - torch.matmul(Ds, patches_sparse_coding)



            coherence = torch.matmul(D_L_w_e.t(), r)


            coherence_conditions = coherence.abs().squeeze(dim=2).max(dim=1)[0] > 1e-4
            true_conditions = (torch.norm(r, dim=1).squeeze(dim=1) > epsilon)*(coherence_conditions)
            false_conditions = ((torch.norm(r, dim=1).squeeze(dim=1) <= epsilon)+(coherence_conditions==False))>0


            if (self.early_stopping and self.extra_input and (self.extra_input_type == 'sparsity')):
                true_conditions = (true_cardinality!=(k+1))
                false_conditions = (true_cardinality==(k+1))



            # if (not (true_conditions != false_conditions).all()):
            #     print(true_conditions.shape)
            #     print(false_conditions.shape)
            #     print(torch.nonzero((true_conditions != false_conditions).int()))
            #     indecesss = torch.arange(true_conditions.numel())[true_conditions == false_conditions]
            #     print(indecesss.numel())
            #     print(indecesss.shape)
            #     print('epsilon=', epsilon)
            #     print('r_prev=', r_prev[:100])
            #     print('self.D_L_w_e=', self.D_L_w_e)
            #     print('D_L_w_e.t()=', D_L_w_e.t())
            #     print('coherence=', coherence[:100])
            #     print('tmp=', tmp[:100])
            #     print('tmp2=', tmp2[:100])
            #     # tmp = torch.norm(r, dim=1).squeeze(dim=1)
            #     # print(tmp.shape)
            #     # print('tmp=',tmp[:100])
            #     # print(true_conditions[:100])
            #     # print(false_conditions[:100])
            #     # print('k=',k)
            #     print('patches', patches[:100])
            #     # print('patches_sparse_coding',patches_sparse_coding[:100])
            #     print('Ds[0]=', Ds[0])
            #     print('r=', r[:100])

            assert(true_conditions != false_conditions).all()



            if (self.extra_info_flag):
                denoised_patches_sparsity[remained_patches] = (k + 1)

            if (self.apply_attention_mode):
                assert(self.enable_rain_net == False)
                if(self.early_stopping):
                    tmp_res[remained_patches, :, k] = torch.matmul(Ds2, patches_sparse_coding)[:, :, 0]
                else:
                    tmp_res[:, :, k] = torch.matmul(Ds2, patches_sparse_coding)[:, :, 0]


            if((patches[false_conditions].numel()>0) and (self.apply_attention_mode == False)):

                if (self.extra_info_flag):
                    denoised_patches_sparsity[remained_patches[true_conditions]] = (k + 1)

                tmp_tmp_sparse_code = patches_sparse_coding[false_conditions].clone()
                tmp_tmp_Ds2 = Ds2[false_conditions]

                if (self.enable_rain_net):
                    tmp_patches_data_atoms = patches_data_atoms[false_conditions]
                    tmp_tmp_rain_sparse_code = torch.mul(tmp_tmp_sparse_code,1-tmp_patches_data_atoms)
                    rain_component_patches[remained_patches[false_conditions]]= torch.matmul(tmp_tmp_Ds2,
                                                                                             tmp_tmp_rain_sparse_code
                                                                            )
                    tmp_tmp_sparse_code = torch.mul(tmp_tmp_sparse_code,tmp_patches_data_atoms)


                denoised_patches[remained_patches[false_conditions]] = torch.matmul(tmp_tmp_Ds2,
                                                                                    tmp_tmp_sparse_code
                                                                            )

            ###################################################################




            # print('**********')
            if(self.early_stopping):
                remained_patches = remained_patches[true_conditions]
                patches = patches[true_conditions]
                Ds = Ds[true_conditions]
                Ds2 = Ds2[true_conditions]
                patches_sparse_coding = patches_sparse_coding[true_conditions]
                if (self.cholesky_decomposition):
                    L = L[true_conditions]
                    correlation_array = correlation_array[true_conditions]
                coherence = coherence[true_conditions]
                if (self.enable_rain_net):
                    patches_data_atoms = patches_data_atoms[true_conditions]

            if (self.early_stopping and self.extra_input and (self.extra_input_type == 'stopping_thresholds')):
                epsilon = epsilon[true_conditions]

            if (self.early_stopping and self.extra_input and (self.extra_input_type == 'sparsity')):
                true_cardinality  = true_cardinality[true_conditions]

            if(remained_patches.numel()==0):
                break

        if (self.apply_attention_mode):
            attention_weights = self.attention_net(original_patches - tmp_res)
            denoised_patches = torch.mul(tmp_res, attention_weights).sum(dim=2, keepdim=True)

        if ((remained_patches.numel() > 0) and (self.apply_attention_mode==False)):
            if (self.enable_rain_net):
                tmp_tmp_rain_sparse_code = torch.mul(patches_sparse_coding,1-patches_data_atoms)
                rain_component_patches[remained_patches] = torch.matmul(Ds2,tmp_tmp_rain_sparse_code)
                patches_sparse_coding = torch.mul(patches_sparse_coding,patches_data_atoms)

            denoised_patches[remained_patches] = torch.matmul(Ds2, patches_sparse_coding)



        if (in_conditions.all() == False):
            tmptmp = torch.zeros_like(backup_patches)
            tmptmp[in_conditions] = denoised_patches
            tmptmp[in_conditions==False] = backup_patches[in_conditions==False]
            denoised_patches = tmptmp
            if (self.enable_rain_net):
                tmptmp2 = torch.zeros_like(backup_patches)
                tmptmp2[in_conditions] = rain_component_patches
                tmptmp2[in_conditions == False] = 0
                rain_component_patches = tmptmp2
            if (self.extra_info_flag):
                tmptmptmp = torch.zeros(backup_patches.shape[0],device = patches.device,dtype=patches.dtype)
                tmptmptmp[in_conditions] = denoised_patches_sparsity
                denoised_patches_sparsity = tmptmptmp


        if (self.extra_info_flag):
            res_data = {}
            res_data['sparsity'] = denoised_patches_sparsity
            res_data['output'] = denoised_patches
            res_data['support_probability'] = support_probability
            res_data['chosen_support'] = chosen_support
            if(self.enable_rain_net):
                res_data['rain_component_patches'] = rain_component_patches

            denoised_patches = res_data



        return denoised_patches

class Attention_Net(nn.Module):
    def __init__(self,n,k,layers=4):
        super(Attention_Net, self).__init__()
        self.model_list1 = nn.ModuleList(
            [nn.Linear(n, n, False ) for i in range(layers)])
        # self.model_list1 = nn.ModuleList(
        #     [nn.Linear(n, n, True) for i in range(layers)])
        self.model_list2 = nn.ModuleList(
            [nn.Linear(k, k, True) for i in range(layers)])

        self.last_linear = nn.Linear(n, 1, False )
        self.softmax = nn.Softmax(dim=1)

        # self.batch_norm1 = nn.BatchNorm1d(n)
        # self.batch_norm2 = nn.BatchNorm1d(k)
        # self.batch_norm3 = nn.BatchNorm1d(n)


    def forward(self,x):
        dim1,dim2= 1,2
        if(len(x.shape)==4):
            dim1, dim2 = 1, 3
        patches = x.transpose(dim1,dim2)
        for i in range(len(self.model_list1)):
            linear1 = self.model_list1[i]
            linear2 = self.model_list2[i]
            tmp = linear2(linear1(patches).transpose(dim1,dim2))
            patches = F.relu(tmp).transpose(dim1,dim2)

            # if(i==0):
            #     patches = self.batch_norm1(patches.transpose(dim1,dim2)).transpose(dim1,dim2)
            #
            # if (i == 1):
            #     patches = self.batch_norm2(patches)
            #
            # if (i == 2):
            #     patches = self.batch_norm3(patches.transpose(dim1,dim2)).transpose(dim1,dim2)

        # print (patches.shape)
        res = self.softmax(self.last_linear(patches)).transpose(dim1,dim2)
        # print (res.shape)
        return res

class Rand_Inner_Patch_OMP_Model(Inner_Patch_OMP_Model):

    def aux_init(self):
        # self.MSE_iterations = 5
        self.MSE_iterations = 2
        n_row, n_col = self.n_row,self.n_col
        print('n_row=',n_row)
        print('n_col=',n_col)
        if(self.apply_attention_mode):
            self.merge_net = Attention_Net(n_row*n_row,self.MSE_iterations+1,3)


    def forward(self, x):

        # representation_map = x['representation'].squeeze(dim=2).abs() > 1e-5
        # rep_ind = torch.nonzero(representation_map)[:,1].view(representation_map.shape[0],-1)
        # print('oracle representation=',rep_ind[0])
        # x = x['input']
        # D = self.D_L_w_e
        # # print(rep_ind.shape)
        # Ds_oracle = D.t()[rep_ind,:].transpose(1,2)
        # rep_oracle = self.get_support_ls_2(x,Ds_oracle)
        # oracle_res = torch.matmul(Ds_oracle,rep_oracle)


        if(self.extra_input):
            # res = torch.zeros_like(x['input'])
            res = torch.zeros(x['input'].shape[0],x['input'].shape[1],self.MSE_iterations+1,dtype=x['input'].dtype,
                              device=x['input'].device)
        else:
            # res = torch.zeros_like(x)
            res = torch.zeros(x.shape[0], x.shape[1], self.MSE_iterations+1, dtype=x.dtype,
                              device=x.device)
        # weight_sum = 0
        if (self.extra_info_flag):
            # sparsity = 0
            support_lst = []

        self.set_rand_OMP(False)
        tmp = Inner_Patch_OMP_Model.forward(self, x)

        if (self.extra_info_flag):
            # current_weight = tmp['support_probability'].view(-1, 1, 1)
            current_res = tmp['output']
            # sparsity += tmp['sparsity']
            support_lst.append(tmp['chosen_support'])

        else:
            current_res = tmp

        res[:, :, 0] = current_res[:, :, 0]

        # oracle_weight = current_weight
        # res += oracle_res*oracle_weight
        # weight_sum += oracle_weight
        # sparsity += representation_map.sum(dim=1).float()

        self.set_rand_OMP(True)
        for i in range(self.MSE_iterations):
            if(self.extra_info_flag):
                tmp = Inner_Patch_OMP_Model.forward(self, x)
                # current_weight = tmp['support_probability'].view(-1,1,1)
                current_res = tmp['output']
                support_lst.append(tmp['chosen_support'])
            else:
                # assert False
                current_res = Inner_Patch_OMP_Model.forward(self, x)
                # current_res = tmp['output'] * current_weight

            res[:,:,i+1]=current_res[:,:,0]
            # res = current_res
        # res /= MSE_iterations
        # res /= weight_sum
        if(self.apply_attention_mode):
            if(self.extra_input):
                ppp = x['input']
            else:
                ppp = x

            weights = self.merge_net(ppp-res)
            res = torch.mul(res, weights).sum(dim=2, keepdim=True)
        else:
            res = res.mean(dim=2,keepdim=True)

        if (self.extra_info_flag):
            sparsity = torch.cat(tuple(support_lst),dim=1)
            # print(sparsity[0])
            sparsity_res = torch.zeros_like(tmp['sparsity'])
            for j in range(self.D_L_w_e.shape[1]):
                sparsity_res += ((sparsity==j).sum(dim=1)>0 )
            res={'output':res,'sparsity':sparsity_res}
        return res

class OMP_Model(nn.Module):
    def __init__(self,k,n_row,n_col,m,sigma_noise,early_stopping=True,
                 initial_mode='DCT',equal_dictionaries = False,learned_weighted_averaging=False,
                 add_DC_atom=False,dictionary_multiscale=False,
                 dictionary_to_load=None
                 ,channels=1
                 ,apply_attention_mode = False
                 ,enable_rain_net = False
                 ,compress_mode = False
                 ):

        super(OMP_Model, self).__init__()

        self.n_row = n_row
        self.n_col = n_col
        self.m = m


        self.inner_OMP_model = Inner_Patch_OMP_Model(k=k,n_row=n_row,n_col=n_col
                 ,m=m,sigma_noise=sigma_noise,early_stopping=early_stopping,
                 initial_mode=initial_mode,equal_dictionaries = equal_dictionaries,
                 add_DC_atom=add_DC_atom,dictionary_multiscale=dictionary_multiscale,
                 dictionary_to_load=dictionary_to_load
                 ,channels=channels
                 ,apply_attention_mode = apply_attention_mode
                 ,enable_rain_net = enable_rain_net)


        self.available_gpus = torch.cuda.device_count()
        print('available_gpus = ',self.available_gpus)
        if(self.available_gpus>1):
            self.inner_OMP_model = nn.DataParallel(self.inner_OMP_model)

        self.learned_weighted_averaging = learned_weighted_averaging
        if (learned_weighted_averaging):
            # #Grisha
            self.alpha = nn.Parameter(0.25 + torch.zeros(1))
            # self.beta = nn.Parameter(0.001 + torch.zeros(1))


        self.channels = channels
        self.compress_mode = compress_mode


        self.regularization_flag = False

        self.atoms_masking_epoch_counter = -1
        self.single_image_mask=False
        self.evaluation_mode = False

        self.extra_input = False
        self.enable_rain_net = enable_rain_net
        if(enable_rain_net):
            self.merge_net = Merge_Net2(9,kernel_len=5,out_channels=3)

        self.get_extra_info_flag_flag = False



    def set_enable_cholesky_decomposition_flag(self, flag):
        self.inner_OMP_model.set_enable_cholesky_decomposition_flag(flag)


    def show_atoms(self):
        self.inner_OMP_model.show_atoms()

    def get_extra_info_flag(self,flag):
        self.get_extra_info_flag_flag = flag
        self.inner_OMP_model.get_extra_info(flag)

    def set_extra_input_flag(self,flag,type):
        self.extra_input = flag
        self.extra_input_type = type
        self.inner_OMP_model.set_extra_input_flag(flag,type)

    def set_batch_OMP_flag(self,flag):
        self.inner_OMP_model.set_batch_OMP_flag(flag)

    def tune_sigma_coef(self,test_set,overrite_max=False,coeff_range=torch.arange(-0.02,0.02,0.001)):
        original_coef = self.inner_OMP_model.stopping_coef.clone()
        coeff_range = coeff_range + original_coef.item()
        psnr_array = torch.zeros_like(coeff_range)
        loss_array = torch.zeros_like(coeff_range)
        for i in range(coeff_range.numel()):
            self.inner_OMP_model.stopping_coef[:] = coeff_range[i]
            print(self.inner_OMP_model.stopping_coef)
            psnr=0
            loss=0
            inner_itr = 2
            for j in range(inner_itr):
                loss_tmp,tmp2,psnr_tmp = calculate_all_loss_model(self,test_set)
                psnr+=psnr_tmp
                loss+=loss_tmp
            psnr /= inner_itr
            loss /= inner_itr
            psnr_array[i]=psnr
            loss_array[i] = loss


        self.inner_OMP_model.stopping_coef[:] = original_coef
        max_indes = torch.argmax(psnr_array)
        max_coef = coeff_range[max_indes]
        print('old coeff = ',original_coef)
        print('new max coeff = ',max_coef)
        if(overrite_max):
            self.inner_OMP_model.stopping_coef[:] = max_coef

        return (coeff_range.detach().numpy(),psnr_array.detach().numpy(),loss_array.detach().numpy())






    def has_manually_update_parameters(self,epoch):
        return False


    def set_evaluation_mode(self,flag):
        self.evaluation_mode = flag
        self.inner_OMP_model.set_evaluation_mode(flag)


    def mask_params_grads(self):
        pass





    def update_parameters(self):
        pass



    def setRegularization(self,flag,mu=1e-5):
    # def setRegularization(self, flag, mu=2e-4):
        self.regularization_flag=flag
        self.regularization_mu = mu

    def has_regularization(self):
        return self.regularization_flag


    def calcualte_mutual_coherence(self,D):
        D_normlized = normlize_kernel(D)
        res = torch.max(torch.abs(
            torch.matmul(D_normlized.t(), D_normlized) - torch.eye(D_normlized.shape[1]).to(device=D_normlized.device,dtype=D_normlized.dtype)))
        return res

    def calcualte_coherence_mean(self,D):
        D_normlized = normlize_kernel(D)
        res = torch.mean(torch.abs(
            torch.matmul(D_normlized.t(), D_normlized) - torch.eye(D_normlized.shape[1]).to(device=D_normlized.device,dtype=D_normlized.dtype)))
        return res

    def get_regularization(self):



        assert(self.regularization_flag)

        inner_model = self.inner_OMP_model
        if(self.available_gpus >1):
            inner_model = self.inner_OMP_model.module

        # res =  2*self.regularization_mu*(self.calcualte_coherence_mean(self.D_L_w_e)+self.calcualte_coherence_mean(self.D_L_d))

        res = self.regularization_mu * (
                    self.calcualte_mutual_coherence(inner_model.D_L_w_e)+self.calcualte_mutual_coherence(inner_model.D_L_d))
        if (inner_model.dictionary_multiscale):
            res/=2
            res += 0.5*self.regularization_mu * (
                self.calcualte_mutual_coherence(inner_model.D_L_w_e2)+self.calcualte_mutual_coherence(inner_model.D_L_d2))



        return res



    def forward(self, x):

        if(self.extra_input):
            extra_info = x
            x = x['input']

        # original_x = x.clone()
        assert (x.shape[1]==self.channels)


        if(self.compress_mode):
            original_x = x.clone()
            # stride = (self.n_row, self.n_col)
            stride = (round(self.n_row/2), round(self.n_col/2))
            pad_left, padd_right, pad_top,  pad_bottom = (0,(self.n_col - x.shape[3]%self.n_col )%self.n_col ,0,
                                                          (self.n_row - x.shape[2]%self.n_row )%self.n_row)
            x = F.pad(x, (pad_left, padd_right, pad_top,  pad_bottom), 'reflect')
            unfold = nn.Unfold(kernel_size=(self.n_row, self.n_col), stride=stride)
            fold = nn.Fold(output_size=(x.shape[2], x.shape[3]), kernel_size=(self.n_row, self.n_col),
                           stride=stride)

        else:
            unfold = nn.Unfold(kernel_size=(self.n_row, self.n_col))
            fold = nn.Fold(output_size=(x.shape[2], x.shape[3]), kernel_size=(self.n_row, self.n_col))




        if(self.available_gpus <=1):
            add_DC_atom_flag = self.inner_OMP_model.add_DC_atom
        else:
            assert False
            # add_DC_atom_flag = self.inner_OMP_model.module.add_DC_atom


        original_x = x.clone()

        patches = unfold(x)
        if(self.extra_input):
            if (self.extra_input_type == 'stopping_thresholds'):
                extra_stopping_thresholds = (patches-unfold(extra_info['output'])).pow(2).sum(dim=1,keepdim=True).sqrt()



        # if(add_DC_atom_flag==False):
        #     means = torch.mean(patches,dim=1,keepdim=True)
        #     patches -= means



        original_patches = patches.clone()
        patches_shape = patches.shape

        patches = torch.unsqueeze(torch.transpose(patches,1,2).reshape(-1,patches.shape[1]),dim=2)

        if (self.extra_input):
            if (self.extra_input_type == 'stopping_thresholds'):
                extra_stopping_thresholds = extra_stopping_thresholds.transpose(1,2).squeeze(dim=2).view(-1)



        if(self.evaluation_mode==False):

            if (self.extra_input):
                if(self.extra_input_type=='stopping_thresholds'):
                    inner_model_input = {'input':patches,'stopping_thresholds':extra_stopping_thresholds}

            else:
                inner_model_input = patches
                # inner_model_input = {'input': patches, 'non_local_true_dist': true_top_val_l2_sq_distances}

            res = self.inner_OMP_model(inner_model_input)

            if (self.get_extra_info_flag_flag):
                avg_sparsity = res['sparsity']
                if (self.enable_rain_net):
                    rain_component_patches = res['rain_component_patches']
                res = res['output']



            denoised_patches = res

        else:
            denoised_patches = torch.zeros_like(patches)
            if (self.get_extra_info_flag_flag):
                avg_sparsity = torch.zeros(patches.shape[0],dtype=patches.dtype,device=patches.device)
                if(self.enable_rain_net):
                    rain_component_patches = torch.zeros_like(patches)

            numnum = 600000

            if(self.channels>1):
                numnum = 15000

            else:
                if (self.n_row == 12):
                    numnum = 60000

            patches_per_group = round(numnum/patches.shape[2])
            groups = torch.arange(patches.shape[0]).split(patches_per_group)
            for gg in groups:
                if (self.extra_input):
                    if (self.extra_input_type == 'stopping_thresholds'):
                        inner_model_input = {'input':patches[gg],'stopping_thresholds':extra_stopping_thresholds[gg]}
                else:
                    inner_model_input = patches[gg]

                res = self.inner_OMP_model(inner_model_input)
                if (self.get_extra_info_flag_flag):
                    avg_sparsity[gg] = res['sparsity']
                    if(self.enable_rain_net):
                        rain_component_patches[gg] = res['rain_component_patches']

                    res = res['output']

                denoised_patches[gg] = res




        denoised_patches = torch.transpose(torch.squeeze(denoised_patches,dim=2).reshape(patches_shape[0],patches_shape[2],patches_shape[1]),1,2)
        ones = torch.ones_like(x)
        ones_unfolded = unfold(ones)
        if(self.learned_weighted_averaging):
            if(self.channels==1):
                patches_variance = torch.var(original_patches, unbiased=True, dim=1, keepdim=True)
                patches_weights = torch.exp(torch.mul(-self.alpha.abs(), patches_variance))


            if (self.channels == 3):
                assert False
                # tmp_tmp_patches = original_patches.view(original_patches.shape[0],
                #          self.channels,self.n_row,self.n_col,original_patches.shape[2])
                # tmp_patches_weights = torch.zeros_like(tmp_tmp_patches)
                # patches_variance = torch.var(tmp_tmp_patches, unbiased=True, dim=(2,3), keepdim=True)
                # tmp_patches_weights[:] = torch.exp(torch.mul(-self.alpha.abs().view(1,-1,1,1,1), patches_variance))
                # patches_weights = tmp_patches_weights.view(*original_patches.shape)


            denoised_patches = torch.mul(denoised_patches,patches_weights)
            ones_unfolded = torch.mul(ones_unfolded,patches_weights)

        denoised_images_sum = fold(denoised_patches)
        div = fold(ones_unfolded)


        # if (self.learned_weighted_averaging):
        #     # muuuu = self.beta.abs()
        #     denoised_images_sum = denoised_images_sum.clone()+torch.mul(original_shape_rain_weights,x)
        #     div = div.clone() + original_shape_rain_weights


        denoised_images = torch.div(denoised_images_sum,div)

        if(self.get_extra_info_flag_flag):
            avg_sparsity = avg_sparsity.reshape(patches_shape[0],patches_shape[2]).mean(dim=1)
            if (self.enable_rain_net):
                rain_component_patches = rain_component_patches.squeeze(dim=2).reshape(patches_shape[0],patches_shape[2],patches_shape[1]).transpose(1,2)
                rain_ones = torch.ones_like(rain_component_patches)

                rain_component_image = torch.div(fold(rain_component_patches), fold(rain_ones))


        if (self.compress_mode):
            denoised_images = denoised_images[:,:,:original_x.shape[2],:original_x.shape[3]].clone()

        if(self.get_extra_info_flag_flag):
            res = {'output':denoised_images
                # ,'dist':shift_patches_distances
                   ,'sparsity':avg_sparsity}
            if (self.enable_rain_net):
                res['rain_component_image'] = rain_component_image
                res['pre_output'] = denoised_images
                res['pre_output2'] = original_x - rain_component_image
                res['output'] = self.merge_net(torch.cat((original_x, res['pre_output2'], res['pre_output']), dim=1))

        else:
            res = (0, denoised_images)

        return res


class Rain_Attention(nn.Module):
    def __init__(self, n_row, n_col, channels, layers=2):
        super(Rain_Attention, self).__init__()
        self.n_row = n_row
        self.n_col = n_col
        self.channels = channels
        assert (n_row % 2 == 1)
        assert (n_col % 2 == 1)

        # n = n_row*n_col*channels
        # self.model_list1 = nn.ModuleList(
        #     [nn.Linear(n, n, True) for i in range(layers)])
        # self.last_linear = nn.Linear(n, channels, False)
        # self.last_linear = nn.Linear(n, n, False)

        # n2 = round(n/2)
        # n3 = round(n2 / 2)
        # n4 = round(n3 / 2)

        # n =3* n_row*n_col
        # n2 = n
        # n3 = n
        # n4 = n
        # #
        # in_lst = [n,n2,n2,n3,n3,n4]
        # out_lst = [n2,n2,n3,n3,n4,n4]
        # # in_lst = [n]*6
        # # out_lst = [n]*6
        # self.model_list1 = nn.ModuleList(
        #     [nn.Linear(in_lst[i], out_lst[i], True) for i in range(len(in_lst))])
        # # #
        # self.last_linear = nn.Linear(n4, 1, False)
        # self.sig = nn.Sigmoid()
        # self.last_linear = nn.Linear(n4, 1, True)
        # self.sig = Rain_Activation()
        # self.sig = lambda x:x
        # in_channel = [channels,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
        # out_channel = [20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
        # in_channel = [channels,100,100,50,50,50,25,25,25,25,10,10]
        # out_channel = [100,100,50,50,50,25,25,25,25,10,10,10]
        # n_list = [15,13,13,11,11,9,9,7,7,5,5,5]

        in_channel = [3*channels, 25, 25, 25, 25, 25]
        out_channel = [25, 25, 25, 25, 25, 25]
        n_list = [11, 9, 9, 7, 7, 5, 5]
        self.n_list = n_list
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(in_channel[i], out_channel[i], kernel_size=(n_list[i], n_list[i])) for i in
             range(len(in_channel))])
        self.last_conv = nn.Conv2d(out_channel[-1], channels, kernel_size=(n_list[-1], n_list[-1]), bias=False)
        self.skip = nn.Conv2d(in_channel[0],out_channel[-1],kernel_size=(7,7),bias=False)
        self.skip2= nn.Conv2d(in_channel[2],out_channel[-3],kernel_size=(7,7),bias=False)

    def forward(self, x):
        assert (len(x.shape) == 4)
        # unfold = nn.Unfold(kernel_size=(self.n_row, self.n_col))
        # fold = nn.Fold(output_size=(x.shape[2], x.shape[3]), kernel_size=(1, 1))
        original_x = x.clone()

        # x = F.pad(x, (pad_left, padd_right, pad_top, pad_bottom), 'reflect')

        # patches = unfold(x).transpose(1,2)
        # # patches = x.squeeze(dim=2)
        # for i in range(len(self.model_list1)):
        #     linear1 = self.model_list1[i]
        #     patches = F.relu(linear1(patches))

        # # # res = fold(self.last_linear(patches).transpose(1, 2))
        # res = fold(self.sig(self.last_linear(patches)).transpose(1,2)) #.expand(-1,-1,self.channels)
        # res = self.last_linear(patches).unsqueeze(dim=2)
        pd_row = 7 // 2
        pd_col = 7 // 2
        pad_left, padd_right, pad_top, pad_bottom = (pd_col, pd_col, pd_row,
                                                  pd_row)
        skip = self.skip(F.pad(x, (pad_left, padd_right, pad_top, pad_bottom), 'replicate'))

        for i, conv in enumerate(self.conv_layers):
            pd_row = self.n_list[i] // 2
            pd_col = self.n_list[i] // 2
            pad_left, padd_right, pad_top, pad_bottom = (pd_col, pd_col, pd_row,
                                                         pd_row)
            x = F.relu(conv(F.pad(x, (pad_left, padd_right, pad_top, pad_bottom), 'replicate')))

            if(i==2):
                pd_row = 7 // 2
                pd_col = 7 // 2
                pad_left, padd_right, pad_top, pad_bottom = (pd_col, pd_col, pd_row,
                                                          pd_row)
                skip2 = self.skip2(F.pad(x, (pad_left, padd_right, pad_top, pad_bottom), 'replicate'))
            if(i==4):
                x = x.clone()+skip2

        x = x.clone()+ skip
        # res = original_x - self.last_conv(F.pad(x, (pad_left, padd_right, pad_top, pad_bottom), 'replicate'))
        res = self.last_conv(F.pad(x, (pad_left, padd_right, pad_top, pad_bottom), 'replicate'))
        # res = self.sig(self.last_conv(F.pad(x, (pad_left, padd_right, pad_top, pad_bottom), 'replicate')))

        # x = F.pad(x, (pad_left, padd_right, pad_top, pad_bottom), 'reflect')
        # patches = unfold(x).transpose(1,2)
        # # patches = x.squeeze(dim=2)
        # for i in range(len(self.model_list1)):
        #     linear1 = self.model_list1[i]
        #     patches = F.relu(linear1(patches))

        # res = fold(self.sig(self.last_linear(patches)).transpose(1,2))

        return res

class Merge_Net2(nn.Module):

    def __init__(self,channels,kernel_len=5,layers=3,out_channels=3):
        super(Merge_Net2, self).__init__()
        assert (kernel_len%2 == 1)
        self.channels = channels
        self.kernel_len = kernel_len
        self.layers = layers
        # in_len = [channels * (kernel_len ** 2)]*layers
        # out_len = [channels * (kernel_len ** 2)]*(layers-1) +[out_channels]
        n = channels * (kernel_len ** 2)
        n2 = 50
        n3 = 25
        in_len = [n,n2,n2,n3]
        out_len = [n2,n2,n3,n3]
        self.model_list = nn.ModuleList([nn.Linear(in_len[i],out_len[i],True) for i in range(len(in_len))])
        self.final_layer = nn.Linear(out_len[-1],out_channels,False)
        self.res_layer = nn.Linear(in_len[0], out_len[-1], False)


    def forward(self,x):
        assert (len(x.shape)==4)
        assert (x.shape[1]==self.channels)


        unfold = nn.Unfold(kernel_size=(self.kernel_len, self.kernel_len),)
        fold = nn.Fold(output_size=(x.shape[2], x.shape[3]), kernel_size=(1, 1))
        pd = self.kernel_len//2
        pad_left, padd_right, pad_top, pad_bottom = (pd, pd, pd, pd)
        new_x = F.pad(x, (pad_left, padd_right, pad_top, pad_bottom), 'replicate')

        patches = unfold(new_x).transpose(1,2)
        # original_patches = patches.clone()
        res_connection = self.res_layer(patches)
        for index,model in enumerate(self.model_list):
            patches = model(patches)
            patches = F.relu(patches)


        patches = patches.clone() + res_connection
        patches = self.final_layer(patches)
        merged_images = fold(patches.transpose(1,2))
        return merged_images

class Merge_Net(nn.Module):

    def __init__(self,channels,kernel_len=5,layers=4):
        super(Merge_Net, self).__init__()
        assert (kernel_len%2 == 1)
        self.channels = channels
        self.kernel_len = kernel_len
        self.layers = layers
        in_len = [channels * (kernel_len ** 2)]*layers
        out_len = [channels * (kernel_len ** 2)]*(layers-1) +[1]
        self.model_list = nn.ModuleList([nn.Linear(in_len[i],out_len[i],False if (i==(layers-1)) else True) for i in range(layers)])
        self.res_layer1 = nn.Linear(in_len[0],out_len[-1],False)
        self.res_layer2 = nn.Linear(in_len[0], out_len[-1], False)

    def forward(self,x):
        assert (len(x.shape)==4)
        assert (x.shape[1]==self.channels)


        unfold = nn.Unfold(kernel_size=(self.kernel_len, self.kernel_len),)
        fold = nn.Fold(output_size=(x.shape[2], x.shape[3]), kernel_size=(1, 1))
        pd = self.kernel_len//2
        pad_left, padd_right, pad_top, pad_bottom = (pd, pd, pd, pd)
        new_x = F.pad(x, (pad_left, padd_right, pad_top, pad_bottom), 'reflect')

        patches = unfold(new_x).transpose(1,2)
        original_patches = patches.clone()
        for index,model in enumerate(self.model_list):
            patches=model(patches)
            if(index != (len(self.model_list)-1)):
                patches = F.relu(patches)
            if(index==1):
                patches = patches.clone() + self.res_layer1(original_patches)

        patches = patches.clone() + self.res_layer2(original_patches)
        merged_images = fold(patches.transpose(1,2))
        return (0,merged_images)


class OMP_Model_Global_MultiScale(nn.Module):

    def __init__(self,sigma_noise,early_stopping=False,
                 initial_mode='DCT',learned_weighted_averaging = False,
                 equal_dictionaries = False,
                 add_DC_atom=False,apply_attention_mode = False):

        super(OMP_Model_Global_MultiScale, self).__init__()
        #nn.ModuleList
        n_list = [8,12]
        k_list = [10,20]
        self.model_list = nn.ModuleList([OMP_Model(k=k,n_row=n,n_col=n,m=4*(n**2),sigma_noise=sigma_noise
                   ,early_stopping=early_stopping,
                 initial_mode=initial_mode,learned_weighted_averaging=learned_weighted_averaging,
                 equal_dictionaries=equal_dictionaries,
                 add_DC_atom=add_DC_atom,dictionary_multiscale=False
                                         ,apply_attention_mode = apply_attention_mode
                                                   ) for n,k in zip(n_list,k_list)])

        self.merge = Merge_Net(len(n_list))
        self.get_extra_info_flag_flag = False


    def get_extra_info_flag(self,flag):
        self.get_extra_info_flag_flag = flag


    def set_enable_cholesky_decomposition_flag(self,flag):
        for model in self.model_list:
            model.set_enable_cholesky_decomposition_flag(flag)

    def set_evaluation_mode(self, flag):
        for model in self.model_list:
            model.set_evaluation_mode(flag)


    def tune_sigma_coef(self,test_set,overrite_max=False,coeff_range=torch.arange(-0.02,0.02,0.001)):
        original_coef = self.inner_OMP_model.stopping_coef.clone()
        coeff_range = coeff_range + original_coef.item()
        psnr_array = torch.zeros_like(coeff_range)
        loss_array = torch.zeros_like(coeff_range)
        for i in range(coeff_range.numel()):
            self.inner_OMP_model.stopping_coef[:] = coeff_range[i]
            print(self.stopping_coef)
            psnr=0
            loss=0
            inner_itr = 2
            for j in range(inner_itr):
                loss_tmp,tmp2,psnr_tmp = calculate_all_loss_model(self,test_set)
                psnr+=psnr_tmp
                loss+=loss_tmp
            psnr /= inner_itr
            loss /= inner_itr
            psnr_array[i]=psnr
            loss_array[i] = loss


        self.inner_OMP_model.stopping_coef[:] = original_coef
        max_indes = torch.argmax(psnr_array)
        max_coef = coeff_range[max_indes]
        print('old coeff = ',original_coef)
        print('new max coeff = ',max_coef)
        if(overrite_max):
            self.inner_OMP_model.stopping_coef[:] = max_coef

        return (coeff_range.detach().numpy(),psnr_array.detach().numpy(),loss_array.detach().numpy())






    def has_manually_update_parameters(self,epoch):
        return bool(sum([model.has_manually_update_parameters(epoch) for model in self.model_list]))

    def mask_params_grads(self):
        for model in self.model_list:
            model.mask_params_grads()





    def setRegularization(self,flag,mu=1e-5):
    # def setRegularization(self, flag, mu=2e-4):
        for model in self.model_list:
            model.setRegularization(flag,mu)

    def has_regularization(self):
        return bool(sum([model.has_regularization() for model in self.model_list]))


    def calcualte_mutual_coherence(self,D):
        return self.model_list[0].calcualte_mutual_coherence(D)

    def calcualte_coherence_mean(self,D):
        return self.model_list[0].calcualte_coherence_mean(D)

    def get_regularization(self):
        return sum([model.get_regularization() for model in self.model_list])/len(self.model_list)




    def forward(self, x):
        assert (x.shape[1]==1)
        if(self.get_extra_info_flag_flag):
            res = {}
            iii = 0
        # denoised_image = torch.zeros_like(x)
        denoised_image = torch.zeros(x.shape[0],len(self.model_list),x.shape[2],x.shape[3],dtype = x.dtype,device=x.device)
        for index,model in enumerate(self.model_list):
            tmp,tmp_denoised = model(x)
            # denoised_image = torch.cat((denoised_image, tmp_denoised), dim=1)
            denoised_image[:,index,:,:] = tmp_denoised[:,0,:,:]
            if (self.get_extra_info_flag_flag):
                res['pre_output'+(str(iii+1) if (iii>0) else '')] = tmp_denoised
                iii += 1

        tmp, denoised_image = self.merge(denoised_image)

        if (self.get_extra_info_flag_flag):
            res['output'] = denoised_image
            return res


        return (0, denoised_image)



class Inner_Patch_SP_Model(nn.Module):
    def __init__(self,k,n_row,n_col,m,sigma_noise,early_stopping=True,
                 initial_mode='DCT',equal_dictionaries = False,dictionary_to_load=None
                 ):
        super(Inner_Patch_SP_Model, self).__init__()
        assert((initial_mode=='random')or(initial_mode=='DCT')or(initial_mode=='load'))
        # self.k=k
        self.register_buffer('k', torch.tensor([k]))
        self.n_row = n_row
        self.n_col = n_col
        self.m = m
        self.sigma_noise = sigma_noise
        self.early_stopping = early_stopping
        self.equal_dictionaries=equal_dictionaries
        self.multi_k = False


        self.regularization_flag = False
        self.extra_info_flag = False
        self.evaluation_mode = False




        initial_dictionary = normlize_kernel(torch.randn(n_row*n_col,m))
        if(initial_mode=='DCT'):
            assert (n_row==n_col)
            # assert (m == 4*n_row*n_col)
            D = normlize_kernel(self.Init_DCT(n_row, int(math.sqrt(m))))
            assert (D.shape == (n_row * n_col, m))
            initial_dictionary = normlize_kernel(D)
            print(initial_dictionary.shape)

        if (initial_mode == 'load'):
            # print(dictionary_to_load.shape)
            initial_dictionary = normlize_kernel(dictionary_to_load.clone())
            assert (initial_dictionary.shape ==(n_row*n_col,m) )



        self.D_L_w_e = nn.Parameter(initial_dictionary.clone())
        self.D_L_d = nn.Parameter(initial_dictionary.clone())



        if (self.early_stopping):
            stopping_coef = 1.085
            self.register_buffer('stopping_coef', torch.tensor([stopping_coef]))




    def set_evaluation_mode(self,flag):
        # self.evaluation_mode = flag
        pass

    def get_extra_info(self,flag):
        self.extra_info_flag = flag

    def setRegularization(self,flag,mu=1e-5):
    # def setRegularization(self, flag, mu=2e-4):
        self.regularization_flag=flag
        self.regularization_mu = mu

    def has_regularization(self):
        return self.regularization_flag

    def get_regularization(self):

        # total_variation_mu =0.0005

        assert(self.regularization_flag)

        res = self.regularization_mu * (
            self.calcualte_mutual_coherence(self.D_L_w_e)+
            (0 if self.equal_dictionaries else self.calcualte_mutual_coherence(self.D_L_d))
        )

        return res

    #taken from: https://github.com/meyerscetbon/Deep-K-SVD/blob/master/Deep_KSVD.py
    def Init_DCT(self,n, m):
        """ Compute the Overcomplete Discrete Cosinus Transform. """
        Dictionary = np.zeros((n, m))
        for k in range(m):
            V = np.cos(np.arange(0, n) * k * np.pi / m)
            if k > 0:
                V = V - np.mean(V)
            Dictionary[:, k] = V / np.linalg.norm(V)
        Dictionary = np.kron(Dictionary, Dictionary)
        Dictionary = Dictionary.dot(np.diag(1 / np.sqrt(np.sum(Dictionary ** 2, axis=0))))
        idx = np.arange(0, n ** 2)
        idx = idx.reshape(n, n, order="F")
        idx = idx.reshape(n ** 2, order="C")
        Dictionary = Dictionary[idx, :]
        Dictionary = torch.from_numpy(Dictionary).double()
        return Dictionary



    def has_manually_update_parameters(self,epoch):
        # if(epoch == 100):
        #     print('SP: updating stopping creteria')
        #     self.sigma_noise = 0.3e-1
        return False


    def mask_params_grads(self):
        return False




    def calcualte_mutual_coherence(self,D):
        D_normlized = normlize_kernel(D)
        res = torch.max(torch.abs(
            torch.matmul(D_normlized.t(), D_normlized) - torch.eye(D_normlized.shape[1]).to(device=D_normlized.device,dtype=D_normlized.dtype)))
        return res



    def get_support_ls_2(self,patches,Ds):
        assert (len(Ds.shape) == 3)

        Ds_t = torch.transpose(Ds,1,2)
        Ds_t_Ds = torch.matmul(Ds_t,Ds)

        D_t_b = torch.matmul(Ds_t,patches)


        new_patches_sparse_coding,LU = torch.solve(D_t_b,Ds_t_Ds)


        return new_patches_sparse_coding




    def computational_SP_step(self,patches,k,iterations = 30):
        D_L_w_e = normlize_kernel(self.D_L_w_e)
        D_L_w_d = self.D_L_w_e
        # D_norm = D_L_w_d.norm(dim=0).view(1, -1, 1)
        if (self.equal_dictionaries):
            D_L_d = D_L_w_d
        else:
            D_L_d = self.D_L_d

        remained_patches = torch.arange(patches.shape[0])
        # denoised_patches = torch.zeros_like(patches)
        # patches_sparsity = torch.zeros(patches.shape[0])
        r = patches
        coherence = torch.matmul(D_L_w_e.t(), r)
        max_func = Pass_Max_Coefficients_k.apply
        tmp = max_func(coherence, k, True)
        tmp = torch.abs(tmp)
        max_vals, max_indices = torch.max(tmp, dim=2, keepdim=True)
        tmp = torch.div(tmp, max_vals)

        Ds_k = torch.transpose(torch.squeeze(torch.matmul(D_L_w_d, tmp), dim=3), 1, 2)
        final_Ds_k = torch.transpose(torch.squeeze(torch.matmul(D_L_d, tmp), dim=3), 1, 2)

        res_Ds_k = Ds_k.clone()
        res_final_Ds_k = final_Ds_k.clone()
        patches_sparse_coding = self.get_support_ls_2(patches, Ds_k)
        res_patches_sparse_coding = patches_sparse_coding.clone()
        r = patches - torch.matmul(Ds_k, patches_sparse_coding)

        tmp = None
        # -------------------------------------------------------------
        for i in range(iterations):  # TODO: may need tp change k here

            # print(i)

            # conditions = (torch.norm(r,dim=1,keepdim=True) >= epsilon)
            r_norms = torch.norm(r, dim=1).squeeze(dim=1)
            coherence = torch.matmul(D_L_w_e.t(), r)
            max_func = Pass_Max_Coefficients_k.apply
            tmp = max_func(coherence, k, True)
            tmp = torch.abs(tmp)
            max_vals, max_indices = torch.max(tmp, dim=2, keepdim=True)
            tmp = torch.div(tmp, max_vals)

            Ds_k_new = torch.transpose(torch.squeeze(torch.matmul(D_L_w_d, tmp), dim=3), 1, 2)
            final_Ds_k_new = torch.transpose(torch.squeeze(torch.matmul(D_L_d, tmp), dim=3), 1, 2)
            Ds_2k = torch.cat((Ds_k, Ds_k_new), dim=2)
            final_Ds_2k = torch.cat((final_Ds_k, final_Ds_k_new), dim=2)

            if (Ds_2k.shape[2] > k):
                tmp = None
                patches_sparse_coding_2k = self.get_support_ls_2(patches, Ds_2k)

                max_func2 = Pass_Max_Coefficients_k.apply
                # tmp2 = max_func2(patches_sparse_coding_2k,  k, False)
                # tmp2 = max_func2(patches_sparse_coding_2k, k, True)
                tmp2 = max_func2(patches_sparse_coding_2k * Ds_2k.norm(dim=1).unsqueeze(dim=2), k, True)
                # tmp2 = max_func2(patches_sparse_coding_2k * Ds_2k.norm(dim=1).unsqueeze(dim=2), k, False)
                tmp2 = torch.abs(tmp2)
                max_vals2, max_indices2 = torch.max(tmp2, dim=2, keepdim=True)
                tmp2 = torch.div(tmp2, max_vals2)

                Ds_k = torch.transpose(torch.squeeze(torch.matmul(torch.unsqueeze(Ds_2k, dim=1), tmp2), dim=3), 1, 2)
                final_Ds_k = torch.transpose(
                    torch.squeeze(torch.matmul(torch.unsqueeze(final_Ds_2k, dim=1), tmp2), dim=3), 1, 2)

            else:
                Ds_k = Ds_2k.clone()
                final_Ds_k = final_Ds_2k.clone()

            patches_sparse_coding = self.get_support_ls_2(patches, Ds_k)
            r = patches - torch.matmul(Ds_k, patches_sparse_coding)
            true_conditions = (torch.norm(r, dim=1).squeeze(dim=1) < r_norms)
            # true_conditions = (torch.norm(r, dim=1).squeeze(dim=1) <= r_norms)

            # print('i = ',i)
            # print('remained_patches num = ',remained_patches.numel(),', from ',res_Ds_k.shape[0])
            if (remained_patches[true_conditions].numel() == 0):
                break

            res_Ds_k[remained_patches[true_conditions]] = Ds_k[true_conditions]
            res_final_Ds_k[remained_patches[true_conditions]] = final_Ds_k[true_conditions]
            res_patches_sparse_coding[remained_patches[true_conditions]] = patches_sparse_coding[true_conditions]

            remained_patches = remained_patches[true_conditions]
            patches = patches[true_conditions]
            Ds_k = Ds_k[true_conditions]
            final_Ds_k = final_Ds_k[true_conditions]
            patches_sparse_coding = patches_sparse_coding[true_conditions]

            r = patches - torch.matmul(Ds_k, patches_sparse_coding)

        return (res_Ds_k, res_final_Ds_k, res_patches_sparse_coding)

    def evaluation_SP_step(self,patches,k,iterations = 30):
        D_L_w_e = normlize_kernel(self.D_L_w_e)
        # D_L_w_d = self.D_L_w_e
        D_L_w_d = D_L_w_e
        D_L_d = D_L_w_e
        # D_norm = D_L_w_d.norm(dim=0).view(1, -1, 1)
        # if (self.equal_dictionaries):
        #     D_L_d = D_L_w_d
        # else:
        #     D_L_d = self.D_L_d

        remained_patches = torch.arange(patches.shape[0])
        # indecis = torch.zeros(patches.shape[0],k,device = patches.device,dtype=int64)

        r = patches
        coherence = torch.matmul(D_L_w_e.t(), r).abs()
        indecis = torch.topk(coherence.squeeze(dim=2),k,dim=1)[1]

        Ds_k = D_L_w_d[:,indecis].transpose(0,1)
        final_Ds_k = D_L_d[:,indecis].transpose(0,1)

        res_Ds_k = Ds_k.clone()
        res_final_Ds_k = final_Ds_k.clone()

        patches_sparse_coding = self.get_support_ls_2(patches, Ds_k)
        res_patches_sparse_coding = patches_sparse_coding.clone()
        r = patches - torch.matmul(Ds_k, patches_sparse_coding)


        # -------------------------------------------------------------
        for i in range(iterations):  # TODO: may need tp change k here


            r_norms = torch.norm(r, dim=1).squeeze(dim=1)
            coherence = torch.matmul(D_L_w_e.t(), r).abs()
            tmp_indecis = torch.topk(coherence.squeeze(dim=2), k, dim=1)[1]
            indecis = torch.cat((indecis, tmp_indecis), dim=1)
            # Ds_k_new = D_L_w_d[:,tmp_indecis].transpose(0,1)

            Ds_2k = D_L_w_d[:,indecis].transpose(0,1)

            if (Ds_2k.shape[2] > k):
                tmp = None
                patches_sparse_coding_2k = self.get_support_ls_2(patches, Ds_2k)
                global_rep = torch.zeros(patches.shape[0],D_L_w_e.shape[1],device=patches.device,dtype=patches.dtype)

                global_rep[torch.arange(patches.shape[0]).view(-1,1),indecis] = patches_sparse_coding_2k.squeeze(dim=2)
                indecis = torch.topk(global_rep.abs(), k, dim=1)[1]

                Ds_k = D_L_w_d[:, indecis].transpose(0, 1)
                final_Ds_k = D_L_d[:, indecis].transpose(0, 1)

            else:
                Ds_k = Ds_2k.clone()
                final_Ds_k = final_Ds_2k.clone()

            patches_sparse_coding = self.get_support_ls_2(patches, Ds_k)
            r = patches - torch.matmul(Ds_k, patches_sparse_coding)
            true_conditions = (torch.norm(r, dim=1).squeeze(dim=1) < r_norms)
            # true_conditions = (torch.norm(r, dim=1).squeeze(dim=1) <= r_norms)

            # print('i = ',i)
            # print('remained_patches num = ',remained_patches.numel(),', from ',res_Ds_k.shape[0])
            if (remained_patches[true_conditions].numel() == 0):
                break

            res_Ds_k[remained_patches[true_conditions]] = Ds_k[true_conditions]
            res_final_Ds_k[remained_patches[true_conditions]] = final_Ds_k[true_conditions]
            res_patches_sparse_coding[remained_patches[true_conditions]] = patches_sparse_coding[true_conditions]

            remained_patches = remained_patches[true_conditions]
            patches = patches[true_conditions]
            Ds_k = Ds_k[true_conditions]
            final_Ds_k = final_Ds_k[true_conditions]
            patches_sparse_coding = patches_sparse_coding[true_conditions]
            indecis = indecis[true_conditions]

            r = patches - torch.matmul(Ds_k, patches_sparse_coding)

        return (res_Ds_k, res_final_Ds_k, res_patches_sparse_coding)

    def evaluation_SP_step2(self,patches,k,iterations = 30):
        D_L_w_e = normlize_kernel(self.D_L_w_e)
        # D_L_w_d = self.D_L_w_e
        D_L_w_d = D_L_w_e
        D_L_d = D_L_w_e

        res_Ds_k = torch.zeros(patches.shape[0],patches.shape[1],k,device=patches.device,dtype=patches.dtype)
        res_final_Ds_k = torch.zeros_like(res_Ds_k)
        res_patches_sparse_coding = torch.zeros(patches.shape[0],k,1,device=patches.device,dtype=patches.dtype)


        for i in range(patches.shape[0]):
            signal = patches[i]
            r = signal
            correlation = torch.matmul(D_L_w_e.t(),r).abs()
            indecis = torch.topk(correlation.squeeze(dim=1),k,dim=0)[1]
            Ds = D_L_w_d[:,indecis]
            sparse_rep, _ = torch.solve(torch.matmul(Ds.t(),signal), torch.matmul(Ds.t(),Ds))
            # sparse_rep,_ = torch.lstsq(signal,Ds)
            # print(Ds.shape)
            # print(sparse_rep.shape)
            # print(signal.shape)
            # print(torch.matmul(Ds,sparse_rep).shape)
            r = signal - torch.matmul(Ds,sparse_rep)
            r_norm = torch.norm(r)

            res_Ds_k[i] = Ds
            res_final_Ds_k[i] = Ds
            res_patches_sparse_coding[i] = sparse_rep

            for tmp_itr in range(iterations):
                correlation = torch.matmul(D_L_w_e.t(), r).abs()
                tmp_indecis = torch.cat((indecis, torch.topk(correlation.squeeze(dim=1), k, dim=0)[1]))
                D2s=D_L_w_d[:,tmp_indecis]
                sparse_rep2, _ = torch.solve(torch.matmul(D2s.t(),signal), torch.matmul(D2s.t(),D2s))

                global_rep = torch.zeros(D_L_w_e.shape[1], device=patches.device, dtype=patches.dtype)
                global_rep[tmp_indecis] = sparse_rep2[:, 0]

                next_indecis = torch.topk(global_rep.abs(), k, dim=0)[1]
                next_Ds = D_L_w_d[:, next_indecis]
                next_sparse_rep, _ = torch.solve(torch.matmul(next_Ds.t(),signal), torch.matmul(next_Ds.t(),next_Ds))
                next_r = signal - torch.matmul(next_Ds, next_sparse_rep)

                next_norm = torch.norm(next_r)
                if(next_norm >= r_norm):
                    break

                indecis = next_indecis
                Ds = next_Ds
                sparse_rep = next_sparse_rep
                r = next_r
                r_norm = next_norm

                res_Ds_k[i] = Ds
                res_final_Ds_k[i] = Ds
                res_patches_sparse_coding[i] = sparse_rep




        return (res_Ds_k, res_final_Ds_k, res_patches_sparse_coding)

    def SP_step(self,patches,k,iterations = 30):
        if(self.evaluation_mode):
            return self.evaluation_SP_step(patches,k,iterations)
            # return self.evaluation_SP_step2(patches, k, iterations)

        return self.computational_SP_step(patches,k,iterations)



    def forward(self, x):

        patches = x
        denoised_patches = torch.zeros_like(patches)
        remained_patches = torch.arange(patches.shape[0])

        if(self.extra_info_flag):
            denoised_patches_sparsity = torch.zeros(patches.shape[0],device = patches.device,dtype=patches.dtype)



        if(self.multi_k == False):
            Ds, Ds2, patches_sparse_coding = self.SP_step(patches, self.k.item())
            denoised_patches = torch.matmul(Ds2, patches_sparse_coding)
            if (self.extra_info_flag):
                res_data = {}
                denoised_patches_sparsity[:] = self.k.item()
                res_data['sparsity'] = denoised_patches_sparsity
                res_data['output'] = denoised_patches
                denoised_patches = res_data

            return denoised_patches


        epsilon = 0
        if(self.early_stopping):
            coef = self.stopping_coef
            epsilon = self.sigma_noise*round(math.sqrt(self.n_row*self.n_col* patches.shape[2]))*coef


        for k in range(self.k):

            Ds, Ds2,patches_sparse_coding  = self.SP_step(patches,k+1)
            r = patches - torch.matmul(Ds, patches_sparse_coding)

            true_conditions = (torch.norm(r, dim=1).squeeze(dim=1) >= epsilon)
            false_conditions = (torch.norm(r, dim=1).squeeze(dim=1) < epsilon)

            if (not (true_conditions != false_conditions).all()):
                print(true_conditions.shape)
                print(false_conditions.shape)
                print(torch.nonzero((true_conditions != false_conditions).int()))
                indecesss = torch.arange(true_conditions.numel())[true_conditions == false_conditions]
                print(indecesss.numel())
                print(indecesss.shape)
                print('epsilon=', epsilon)
                print('r_prev=', r_prev[:100])
                print('self.D_L_w_e=', self.D_L_w_e)
                print('D_L_w_e.t()=', D_L_w_e.t())
                print('coherence=', coherence[:100])
                print('tmp=', tmp[:100])
                print('tmp2=', tmp2[:100])
                # tmp = torch.norm(r, dim=1).squeeze(dim=1)
                # print(tmp.shape)
                # print('tmp=',tmp[:100])
                # print(true_conditions[:100])
                # print(false_conditions[:100])
                # print('k=',k)
                print('patches', patches[:100])
                # print('patches_sparse_coding',patches_sparse_coding[:100])
                print('Ds[0]=', Ds[0])
                print('r=', r[:100])

            assert(true_conditions != false_conditions).all()

            if(self.extra_info_flag):
                denoised_patches_sparsity[remained_patches] = (k+1)

            denoised_patches[remained_patches] = torch.matmul(Ds2,patches_sparse_coding)
            remained_patches = remained_patches[true_conditions]
            patches = patches[true_conditions]

            if(remained_patches.numel()==0):
                break

        if (self.extra_info_flag):
            res_data = {}
            res_data['sparsity'] = denoised_patches_sparsity
            res_data['output'] = denoised_patches
            denoised_patches = res_data


        return denoised_patches





class LISTA_Patch_Model(nn.Module):
    #n_row,n_col,m : dictionary dims
    def __init__(self,k,n_row,n_col,m,sigma_noise,different_filter_threshold=False,
                 initial_mode='random',dictionary_to_load = None,mode=1
                 ):
        super(LISTA_Patch_Model, self).__init__()
        self.k=k
        self.n_row = n_row
        self.n_col = n_col
        self.m = m
        self.sigma_noise = sigma_noise
        self.different_filter_threshold= different_filter_threshold
        self.mode = mode
        assert (mode==1 or mode==2)

        self.regularization_flag = False
        self.extra_info_flag = False
        # c = 100
        c = 1000
        if(mode==1):
            if(initial_mode=='random'):
                t = 1 / (m * n_row * n_col)
                initial_dictionary = math.sqrt(t) * (2 * torch.rand(n_row * n_col, m) - 1)
                self.D_L_w_e = nn.Parameter((1 / c) * initial_dictionary.clone())
                self.D_L_w_d = nn.Parameter(initial_dictionary.clone())
                self.D_L_d = nn.Parameter(initial_dictionary.clone())

            if (initial_mode == 'random_n'):
                initial_dictionary = torch.randn(n_row * n_col, m)
                self.D_L_w_e = nn.Parameter((1 / c) * initial_dictionary.clone())
                self.D_L_w_d = nn.Parameter(initial_dictionary.clone())
                self.D_L_d = nn.Parameter(initial_dictionary.clone())

            if (initial_mode == 'load'):
                initial_dictionary = dictionary_to_load.clone()
                assert (initial_dictionary.shape == (n_row * n_col, m))
                self.D_L_w_e = nn.Parameter((1 / c) * initial_dictionary.clone())
                self.D_L_w_d = nn.Parameter(initial_dictionary.clone())
                self.D_L_d = nn.Parameter(initial_dictionary.clone())

        if (mode == 2):
            if (initial_mode == 'random'):
                t = 1 / (m * n_row * n_col)
                initial_dictionary = math.sqrt(t) * (2 * torch.rand(n_row * n_col, m) - 1)
                self.Q = nn.Parameter(torch.eye(initial_dictionary.shape[1], dtype=initial_dictionary.dtype,
                                                device=initial_dictionary.device) - (1 / c) * torch.matmul(
                    initial_dictionary.t(), initial_dictionary))

                self.W = nn.Parameter((1/c)* initial_dictionary.clone().t())
                self.D_L_d = nn.Parameter(initial_dictionary.clone())

            if (initial_mode == 'random_n'):
                initial_dictionary = torch.randn(n_row * n_col, m)
                self.Q = nn.Parameter(torch.eye(initial_dictionary.shape[1], dtype=initial_dictionary.dtype,
                                                device=initial_dictionary.device) - (1 / c) * torch.matmul(
                    initial_dictionary.t(), initial_dictionary))

                self.W = nn.Parameter((1 / c) * initial_dictionary.clone().t())
                self.D_L_d = nn.Parameter(initial_dictionary.clone())

            if (initial_mode == 'load'):
                initial_dictionary = dictionary_to_load.clone()
                assert (initial_dictionary.shape == (n_row * n_col, m))
                self.Q = nn.Parameter(torch.eye(initial_dictionary.shape[1], dtype=initial_dictionary.dtype,
                                                device=initial_dictionary.device) - (1 / c) * torch.matmul(
                    initial_dictionary.t(), initial_dictionary))

                self.W = nn.Parameter((1 / c) * initial_dictionary.clone().t())
                self.D_L_d = nn.Parameter(initial_dictionary.clone())

        # self.D_L_w_d = nn.Parameter(initial_dictionary)
        # self.D_L_d = nn.Parameter(initial_dictionary)
        if (different_filter_threshold):
            self.Threshold = nn.Parameter((1 / c) + torch.zeros(initial_dictionary.shape[1]))
        else:

            self.Threshold = nn.Parameter((1 / c) + torch.zeros(1))



    def set_evaluation_mode(self,flag):
        pass

    def get_extra_info(self,flag):
        self.extra_info_flag = flag

    def has_manually_update_parameters(self, epoch):
        return False

    def setRegularization(self,flag,mu=1e-5):
    # def setRegularization(self, flag, mu=2e-4):
        self.regularization_flag=flag
        self.regularization_mu = mu

    def has_regularization(self):
        return self.regularization_flag

    def get_regularization(self):

        # total_variation_mu =0.0005

        assert(self.regularization_flag)

        res = self.regularization_mu * (
            self.calcualte_mutual_coherence(self.D_L_w_e)+
            self.calcualte_mutual_coherence(self.D_L_w_d)
        )

        return res

    def calcualte_mutual_coherence(self,D):
        D_normlized = normlize_kernel(D)
        res = torch.max(torch.abs(
            torch.matmul(D_normlized.t(), D_normlized) - torch.eye(D_normlized.shape[1]).to(device=D_normlized.device,dtype=D_normlized.dtype)))
        return res


    def forward(self, x):
        assert (len(x.shape)==3)

        if(self.mode==1):
            D_L_w_e = self.D_L_w_e
            D_L_w_d = self.D_L_w_d
            D_L_d = self.D_L_d
            patches = x
            patches_sparse_coding = torch.zeros(patches.shape[0],D_L_w_e.shape[1],patches.shape[2],dtype=patches.dtype,
                                                device=patches.device)
            for k in range(self.k):
                r = patches - torch.matmul(D_L_w_d, patches_sparse_coding)
                coherence = torch.matmul(D_L_w_e.t(),r)
                patches_sparse_coding = softTHR(patches_sparse_coding+coherence,self.Threshold.view(1,self.Threshold.numel(),1))
        else:
            Q = self.Q
            W = self.W
            D_L_d = self.D_L_d
            patches = x
            patches_sparse_coding = torch.zeros(patches.shape[0], D_L_d.shape[1], patches.shape[2], dtype=patches.dtype,
                                                device=patches.device)
            tmp2 = torch.matmul(W, patches)
            for k in range(self.k):
                tmp1 = torch.matmul(Q,patches_sparse_coding)
                patches_sparse_coding = softTHR(tmp1+tmp2,self.Threshold.view(1,self.Threshold.numel(),1))



        res = torch.matmul(D_L_d,patches_sparse_coding)

        if(self.extra_info_flag):
            dictt = {}
            dictt['output'] = res
            dictt['sparse_coding'] = patches_sparse_coding
            dictt['sparsity'] = (patches_sparse_coding.abs() > 1e-7).squeeze(dim=2).sum(dim=1).to(dtype=patches.dtype)
            res = dictt


        return res



MP_epsilon = 1e-7

