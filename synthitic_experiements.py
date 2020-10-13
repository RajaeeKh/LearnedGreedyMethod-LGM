import Models as models
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import torch.nn as nn
import math
import torch.optim as optim
import sys
import gc
from torch.optim.lr_scheduler import StepLR
import time


# taken from: https://github.com/meyerscetbon/Deep-K-SVD/blob/master/Deep_KSVD.py
def Init_DCT(n, m):
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
    # Dictionary = torch.from_numpy(Dictionary)
    return Dictionary


def create_synthitic_data(n, m, number_of_experiements=100, dictionary='DCT'):
    # data_settings

    a, b = 1e-5, 1
    #multiple cardinality
    # min_sparsity = 5
    # max_sparsity = 11
    
    min_sparsity = 10
    max_sparsity = 11
    saprsity_range = max_sparsity - min_sparsity
    train_signals_per_sparsity = 10000
    test_signals_per_sparsity = 2000

    # train_signals_per_sparsity = 3000000
    # test_signals_per_sparsity = 50000

    assert (dictionary == 'DCT') or (dictionary == 'rand')
    if (dictionary == 'rand'):
        D = np.random.rand(number_of_experiements, n, m)
    if (dictionary == 'DCT'):
        D_tmp = Init_DCT(int(math.sqrt(n)), int(math.sqrt(m)))
        assert (D_tmp.shape == (n, m))
        D = np.zeros((number_of_experiements, n, m))
        for i in range(number_of_experiements):
            D[i] = D_tmp

    train_representation = np.zeros((number_of_experiements, m, train_signals_per_sparsity * saprsity_range))
    train_sparsity_map = np.zeros((number_of_experiements, train_signals_per_sparsity * saprsity_range), dtype='int')
    test_representation = np.zeros((number_of_experiements, m, test_signals_per_sparsity * saprsity_range))
    test_sparsity_map = np.zeros((number_of_experiements, test_signals_per_sparsity * saprsity_range), dtype='int')

    train_signals = np.zeros((number_of_experiements, n, train_signals_per_sparsity * saprsity_range))
    test_signals = np.zeros((number_of_experiements, n, test_signals_per_sparsity * saprsity_range))

    test_representation2 = np.zeros((number_of_experiements, m, test_signals_per_sparsity * saprsity_range))
    test_sparsity_map2 = np.zeros((number_of_experiements, test_signals_per_sparsity * saprsity_range), dtype='int')
    test_signals2 = np.zeros((number_of_experiements, n, test_signals_per_sparsity * saprsity_range))

    for i in range(number_of_experiements):
        for j in range(saprsity_range):
            current_sparsity = j + min_sparsity
            for k in range(train_signals_per_sparsity):
                supp = np.random.permutation(m)[:current_sparsity]
                coef = np.random.rand(current_sparsity) * (b - a) + a
                signs = np.random.randint(2, size=current_sparsity)
                signs[signs == 0] = -1
                train_representation[i, supp, j * train_signals_per_sparsity + k] = np.multiply(coef, signs)
                train_sparsity_map[i, j * train_signals_per_sparsity + k] = current_sparsity

            for k in range(test_signals_per_sparsity):
                supp = np.random.permutation(m)[:current_sparsity]
                coef = np.random.rand(current_sparsity) * (b - a) + a
                signs = np.random.randint(2, size=current_sparsity)
                signs[signs == 0] = -1
                test_representation[i, supp, j * test_signals_per_sparsity + k] = np.multiply(coef, signs)
                test_sparsity_map[i, j * test_signals_per_sparsity + k] = current_sparsity

            for k in range(test_signals_per_sparsity):
                supp = np.random.permutation(m)[:current_sparsity]
                coef = np.random.rand(current_sparsity) * (b - a) + a
                signs = np.random.randint(2, size=current_sparsity)
                signs[signs == 0] = -1
                test_representation2[i, supp, j * test_signals_per_sparsity + k] = np.multiply(coef, signs)
                test_sparsity_map2[i, j * test_signals_per_sparsity + k] = current_sparsity

    for i in range(number_of_experiements):
        print('D[i].shape = ', D[i].shape)
        print('train_representation[i].shape = ', train_representation[i].shape)
        train_signals[i] = np.matmul(D[i], train_representation[i])
        # norm = np.linalg.norm(train_signals[i],axis=0).reshape((1,train_signals[i].shape[1]))
        norm = np.amax(np.abs(train_signals[i]), axis=0).reshape((1, train_signals[i].shape[1]))
        print('norm.shape = ', norm.shape)
        train_signals[i] = np.divide(train_signals[i], norm)
        train_representation[i] = np.divide(train_representation[i], norm)

        test_signals[i] = np.matmul(D[i], test_representation[i])
        # norm = np.linalg.norm(test_signals[i], axis=0).reshape((1, test_signals[i].shape[1]))
        norm = np.amax(np.abs(test_signals[i]), axis=0).reshape((1, test_signals[i].shape[1]))
        print('norm.shape = ', norm.shape)
        test_signals[i] = np.divide(test_signals[i], norm)
        test_representation[i] = np.divide(test_representation[i], norm)

        test_signals2[i] = np.matmul(D[i], test_representation2[i])
        # norm = np.linalg.norm(test_signals2[i], axis=0).reshape((1, test_signals2[i].shape[1]))
        norm = np.amax(np.abs(test_signals2[i]), axis=0).reshape((1, test_signals2[i].shape[1]))
        print('norm.shape = ', norm.shape)
        test_signals2[i] = np.divide(test_signals2[i], norm)
        test_representation2[i] = np.divide(test_representation2[i], norm)

    data = {}
    data['train_representation'] = train_representation
    data['train_sparsity_map'] = train_sparsity_map
    data['test_representation'] = test_representation
    data['test_sparsity_map'] = test_sparsity_map
    data['train_signals'] = train_signals
    data['test_signals'] = test_signals

    data['test_representation2'] = test_representation2
    data['test_sparsity_map2'] = test_sparsity_map2
    data['test_signals2'] = test_signals2

    data['D'] = D
    data['number_of_experiements'] = number_of_experiements
    data['min_sparsity'] = min_sparsity
    data['max_sparsity'] = max_sparsity
    data['test_signals_per_sparsity'] = test_signals_per_sparsity

    return data


class Synthitic_Dataset(Dataset):

    def __init__(self, signals_matrix, representation_matrix, sparsity_vector, D, transform=None):
        assert (len(signals_matrix.shape) == 2)
        self.transform = transform
        self.signals = []
        min_sparsity = np.min(sparsity_vector)
        max_sparsity = np.max(sparsity_vector)
        hist = np.zeros(max_sparsity - min_sparsity + 1)
        for i in range(max_sparsity - min_sparsity + 1):
            hist[i] = np.sum(sparsity_vector == (i + min_sparsity))
        assert (hist == hist[0]).all()
        self.signals_per_sparsity = hist[0]
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        self.D = D
        print(D.shape)
        print(representation_matrix.shape)
        print(signals_matrix.shape)
        assert (np.max(np.abs(np.matmul(D, representation_matrix) - signals_matrix)) < 1e-5)
        for i in range(signals_matrix.shape[1]):
            data_dict = {'signal': signals_matrix[:, i].reshape((-1, 1)),
                         'output': signals_matrix[:, i].reshape((-1, 1)),
                         'representation': representation_matrix[:, i].reshape((-1, 1)),
                         'sparsity': np.array([sparsity_vector[i]])
                         }
            assert (np.sum(np.abs(representation_matrix[:, i]) > 1e-5) == sparsity_vector[i])

            self.signals.append(data_dict)

    def get_sparsity_details(self):
        return {'signals_per_sparsity': self.signals_per_sparsity, 'min_sparsity': self.min_sparsity,
                'max_sparsity': self.max_sparsity}

    def get_true_dictionary(self):
        return self.D

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        sample = signal

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor_Synthitic(object):
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype

    def __call__(self, sample):
        res = {}
        for i in sample.keys():
            if (i != 'sparsity'):
                res[i] = torch.from_numpy(sample[i]).clone().to(device=self.device, dtype=self.dtype)
            else:
                res[i] = torch.from_numpy(sample[i]).clone().to(device=self.device)

        return res


class AddWhiteGaussianNoise_Synthitic(object):
    def __init__(self, sigma, noise=None):
        self.sigma = sigma
        # self.nnnn = noise
        # print('AddWhiteGaussianNoise_Synthitic: Be aware of the noise **********')

    def __call__(self, sample):
        noise = self.sigma * torch.randn_like(sample['signal'])
        # noise = self.sigma *self.nnnn
        # print('***************************=',noise[0,0])
        # while (noise.pow(2).sum() > sample['signal'].numel()*self.sigma**2):
        #   noise = self.sigma * torch.randn_like(sample['signal'])
        sample['signal'] += noise
        return sample


class Calculate_Threshold(object):
    def __call__(self, sample):
        sample['output'] = (sample['signal'] - sample['output']).pow(2).sum().view(1)
        return sample


def calculate_all_loss_model_synthitic_data(model, test_dataset, send_extra_info=False, extra_info_function=None):
    # batch_size = 100
    sparsity_details = test_dataset.get_sparsity_details()
    batch_size = round(sparsity_details['signals_per_sparsity'].item())
    min_sparsity = round(sparsity_details['min_sparsity'].item())
    max_sparsity = round(sparsity_details['max_sparsity'].item())
    groups = round(len(test_dataset) / batch_size)
    res = {}
    with torch.no_grad():
        loss = torch.zeros(groups)
        sparsity = torch.zeros(groups)
        true_sparsity = torch.zeros(groups)
        creterion = nn.MSELoss()
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        model.set_evaluation_mode(True)
        model.get_extra_info(True)
        for i, batch in enumerate(dataloader):
            # print(i)
            sys.stdout.flush()
            signals = batch['signal']
            signals_sprsity = batch['sparsity']
            output = batch['output']
            assert (signals_sprsity == (i + min_sparsity)).all()
            true_sparsity[i] = i + min_sparsity
            if (send_extra_info):
                # signals = {'input':signals,'representation':batch['representation']}
                # model.set_extra_input_flag(True)
                signals = {'input': signals,
                           'stopping_thresholds': (signals - batch['output']).pow(2).sum(dim=(1, 2)).sqrt(),
                           'sparsity': signals_sprsity.view(-1)}
            res = model(signals)
            # print(output.shape)
            # print(res['output'].shape)
            loss[i] = creterion(res['output'], output).item()
            sparsity[i] = res['sparsity'].mean().item()
            torch.cuda.empty_cache()

        model.set_evaluation_mode(False)
        model.get_extra_info(False)
        loss_average = loss.mean().item()
        sparsity_avgerage = sparsity.mean().item()
        true_sparsity = true_sparsity.mean().item()
        sys.stdout.flush()
        res['loss_average'] = loss_average
        res['sparsity_average'] = sparsity_avgerage
        res['true_sparsity'] = true_sparsity
        if (extra_info_function):
            res['extra_info'] = extra_info_function(model)

    # return loss_average,sparsity_avgerage,true_sparsity,loss , sparsity
    return res


def train_model_synthitic_data(train_dataset, test_dataset, model, model_name, criterion, batch_size=10, epochs=200,
                               learning_rate=0.002, optim_type='Adam', extra_input_mode=False, enable_schedluer=False,
                               extra_info_function=None):
    print('starting training ', model_name)
    assert ((optim_type == 'Adam') or (optim_type == 'SGD'))
    print('optimizer=', optim_type)

    if (optim_type == 'Adam'):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if (optim_type == 'SGD'):
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    if (enable_schedluer):
        print('scheduler enabled')
        scheduler = StepLR(optimizer, step_size=50, gamma=0.6)

    # test_loss_array = torch.zeros(epochs + 1)
    # test_sparsity_array = torch.zeros(epochs + 1)
    res_per_epoch = []

    min_MSE = 1e20

    for epoch in range(epochs):
        sys.stdout.flush()
        torch.cuda.empty_cache()
        gc.collect()

        tt = time.time()

        # print_flag = (epoch %100 == 0)
        print_flag = True

        # test_loss_array[epoch], test_sparsity_array[epoch], tmp1,tmp2,tmp3 = calculate_all_loss_model_synthitic_data(
        #         model, test_dataset,send_extra_info=extra_input_mode,extra_info_function=extra_info_function)
        res = calculate_all_loss_model_synthitic_data(
            model, test_dataset, send_extra_info=extra_input_mode, extra_info_function=extra_info_function)

        res_per_epoch.append(res)

        # print('hhhhhhhhhhhhh')
        if (print_flag):
            print('epoch = ', epoch)

            print('test set epoch ', epoch, 'test_loss = ', res['loss_average'], 'test_sparsity = ',
                  res['sparsity_average'])

            sys.stdout.flush()

        if (res['loss_average'] < min_MSE):
            min_MSE = res['loss_average']
            if (print_flag):
                print('updating top model')
            top_data = {
                'epoch': epoch,
                'model_state_dict': dict(
                    [(k, model.state_dict()[k].clone().detach()) for k in model.state_dict().keys()]),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion': criterion
            }
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'criterion': criterion
            # }, 'top_'+model_name + '_model.pt')

        # if (model.has_manually_update_parameters(epoch)):
        #     assert (False)

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for batch in dataloader:
            signals = batch['signal']
            output = batch['output']

            optimizer.zero_grad()

            if (extra_input_mode):
                signals = {'input': signals,
                           'stopping_thresholds': (signals - batch['output']).pow(2).sum(dim=(1, 2)).sqrt(),
                           'sparsity': batch['sparsity'].view(-1)}

            restored_signals = model(signals)
            loss = criterion(output, restored_signals)
            if (model.has_regularization()):
                loss += model.get_regularization()

            # #TODO: remove this
            # loss = torch.log10(loss)

            # print(loss.item())
            sys.stdout.flush()
            torch.cuda.empty_cache()
            loss.backward()

            # model.mask_params_grads()

            optimizer.step()

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion': criterion
            }, model_name + '_model.pt')

        if (enable_schedluer):
            scheduler.step()
        tt = time.time() - tt
        print('time = ', tt)

    res = calculate_all_loss_model_synthitic_data(
        model, test_dataset, send_extra_info=extra_input_mode, extra_info_function=extra_info_function)

    res_per_epoch.append(res)

    print('best epcoh = ', top_data['epoch'])
    print('loading top model')
    model.load_state_dict(top_data['model_state_dict'])

    return res_per_epoch


# {'signal':signals_matrix[:,i].reshape((-1,1)),
#                          'output': signals_matrix[:, i].reshape((-1, 1)),
#                          'representation':representation_matrix[:,i].reshape((-1,1)),
#                          'sparsity':np.array([sparsity_vector[i]])
#                          }

def get_oracle_MSE(testset):
    D = torch.from_numpy(testset.get_true_dictionary())
    creterion = nn.MSELoss()
    MSE = 0
    sparsity = 0
    for i in range(len(testset)):
        sample = testset[i]
        noisy_signal = sample['signal']
        rep = sample['representation']
        supp = rep.abs().view(-1) > 1e-5
        Ds = D.to(device=noisy_signal.device, dtype=noisy_signal.dtype)[:, supp]
        oracle_rep = torch.matmul(torch.pinverse(Ds), noisy_signal)
        oracle_restored_signal = torch.matmul(Ds, oracle_rep)
        MSE += creterion(sample['output'], oracle_restored_signal).item()
        sparsity += sample['sparsity'].item()

    MSE /= len(testset)
    sparsity /= len(testset)
    return MSE, sparsity


def calculate_average_SNR(dataset):
    snr = torch.zeros(len(dataset), dtype=dataset[0]['signal'].dtype, device=dataset[0]['signal'].device)
    for i in range(len(dataset)):
        sample = dataset[i]
        signal_energy = sample['output'].pow(2).sum()
        noise_energy = (sample['signal'] - sample['output']).pow(2).sum()
        snr[i] = 10 * torch.log10(signal_energy / noise_energy)

    print('SNR average = ', snr.mean().item())
    print('SNR variance = ', snr.var().item())

    return snr.mean().item()


def calculate_dict_distance(true_D, approx_D):
    assert (len(true_D.shape) == 2)
    assert (len(approx_D.shape) == 2)
    true_D = models.normlize_kernel(true_D).to(dtype=approx_D.dtype, device=approx_D.device)
    approx_D = models.normlize_kernel(approx_D)
    dist = 0
    for i in range(true_D.shape[1]):
        dist += (1 - torch.matmul(approx_D.t(), true_D[:, i].view(-1, 1)).abs()).min()

    dist /= true_D.shape[1]
    return dist.item()


def get_model_dict_info(true_D, model):
    if (issubclass(type(model), models.Inner_Patch_OMP_Model)):
        return {'D_L_w_e_dist': calculate_dict_distance(true_D, model.D_L_w_e),
                'D_L_d_dist': calculate_dict_distance(true_D, model.D_L_d)}

    if (issubclass(type(model), models.Inner_Patch_SP_Model)):
        # print(calculate_dict_distance(true_D, model.D_L_w_e))
        return {'D_L_w_e_dist': calculate_dict_distance(true_D, model.D_L_w_e),
                'D_L_d_dist': calculate_dict_distance(true_D, model.D_L_d)}

    if (issubclass(type(model), models.Patch_MP_Model)):
        return {'D_L_w_e_dist': calculate_dict_distance(true_D, model.D_L_w_e),
                'D_L_d_dist': calculate_dict_distance(true_D, model.D_L_d)}

    if (issubclass(type(model), models.LISTA_Patch_Model)):
        if (model.mode == 1):
            return {'D_L_w_e_dist': calculate_dict_distance(true_D, model.D_L_w_e),
                    'D_L_w_d_dist': calculate_dict_distance(true_D, model.D_L_w_d),
                    'D_L_d_dist': calculate_dict_distance(true_D, model.D_L_d)}
        return {'D_L_d_dist': calculate_dict_distance(true_D, model.D_L_d)}

    assert False


def run_experiemnts():
    n = 100
    m = 4 * n

    # sigma_noise_range = [0.04,0.06,0.08,0.1,0.12,0.14]
    sigma_noise_range = [0.04]
    MMSE_run_list = [True]

    data = create_synthitic_data(n, m, 1, 'DCT')
    number_of_experiements = data['number_of_experiements']
    max_sparsity = data['max_sparsity']
    min_sparsity = data['min_sparsity']
    sparsity_range = max_sparsity - min_sparsity

    lista_mse_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    lista_sparsity_array = torch.zeros(number_of_experiements, len(sigma_noise_range))

    oracle_mse_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    oracle_sparsity_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    omp_true_dict_mse_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    omp_true_dict_sparsity_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    omp_true_dict2_mse_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    omp_true_dict2_sparsity_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    lista_true_dict_mse_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    lista_true_dict_sparsity_array = torch.zeros(number_of_experiements, len(sigma_noise_range))

    omp_true_dict_MMSE_mse_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    omp_true_dict_MMSE_sparsity_array = torch.zeros(number_of_experiements, len(sigma_noise_range))

    omp_mse_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    omp_sparsity_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    sp_mse_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    sp_sparsity_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    mp_mse_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    mp_sparsity_array = torch.zeros(number_of_experiements, len(sigma_noise_range))

    omp_mse_true_cardinality_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    omp_sparsity_true_cardinality_array = torch.zeros(number_of_experiements, len(sigma_noise_range))

    sp_true_dict_mse_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    sp_true_dict_sparsity_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    mp_true_dict_mse_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    mp_true_dict_sparsity_array = torch.zeros(number_of_experiements, len(sigma_noise_range))

    omp_MMSE_mse_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    omp_MMSE_sparsity_array = torch.zeros(number_of_experiements, len(sigma_noise_range))

    omp_post_training_MMSE_mse_array = torch.zeros(number_of_experiements, len(sigma_noise_range))
    omp_post_training_MMSE_sparsity_array = torch.zeros(number_of_experiements, len(sigma_noise_range))

    true_sparsity_array = torch.zeros(number_of_experiements, len(sigma_noise_range))

    snr_list = []
    training_procedure_data = []
    models_to_save = []

    # t = 1 / (m * n)
    # D = math.sqrt(t) * (2 * np.random.rand(number_of_experiements,n, m) - 1)
    true_D = data['D']
    D = np.random.randn(number_of_experiements, n, m)
    print('true_d.shape=', true_D.shape)
    print('D.shape=', D.shape)
    for i in range(number_of_experiements):
        for j in range(len(sigma_noise_range)):
            print('experiement number ', i)
            print('noise level ', sigma_noise_range[j])
            current_dictionary = torch.from_numpy(D[i])
            sigma_noise = sigma_noise_range[j]
            # # k_omp = 50
            # k_omp = 10
            k_omp = 15

            omp_model = models.Inner_Patch_OMP_Model(k_omp, round(math.sqrt(n)), round(math.sqrt(n)), m, sigma_noise,
                                                     early_stopping=True,
                                                     #  early_stopping=False,
                                                     initial_mode='load', equal_dictionaries=False,
                                                     dictionary_to_load=current_dictionary,
                                                     )

            omp_model.stopping_coef[:] = 1
            omp_model.setRegularization(True, 5e-5)
            omp_model.to(dtype=dtype, device=device)
            # omp_model.set_enable_cholesky_decomposition_flag(True)

            omp_model2 = models.Inner_Patch_OMP_Model(k_omp, round(math.sqrt(n)), round(math.sqrt(n)), m, sigma_noise,
                                                      early_stopping=True,
                                                      initial_mode='load', equal_dictionaries=False,
                                                      dictionary_to_load=current_dictionary
                                                      )
            omp_model2.set_extra_input_flag(True, 'sparsity')
            omp_model2.setRegularization(True, 5e-5)
            omp_model2.to(dtype=dtype, device=device)

            omp_model_true_dict = models.Inner_Patch_OMP_Model(k_omp, round(math.sqrt(n)), round(math.sqrt(n)), m,
                                                               sigma_noise,
                                                               early_stopping=True,
                                                               initial_mode='load', equal_dictionaries=True,
                                                               dictionary_to_load=torch.from_numpy(true_D[i])
                                                               )

            omp_model_true_dict.to(dtype=dtype, device=device)
            omp_model_true_dict.set_evaluation_mode(True)
            omp_model_true_dict.set_extra_input_flag(True, 'sparsity')
            # omp_model_true_dict.set_extra_input_flag(True)

            omp_model_true_dict2 = models.Inner_Patch_OMP_Model(k_omp, round(math.sqrt(n)), round(math.sqrt(n)), m,
                                                                sigma_noise,
                                                                early_stopping=True,
                                                                initial_mode='load', equal_dictionaries=True,
                                                                dictionary_to_load=torch.from_numpy(true_D[i])
                                                                )

            omp_model_true_dict2.to(dtype=dtype, device=device)
            omp_model_true_dict2.set_evaluation_mode(True)
            omp_model_true_dict2.stopping_coef[:] = 1

            mse_omp_model = models.Rand_Inner_Patch_OMP_Model(k_omp, round(math.sqrt(n)), round(math.sqrt(n)), m,
                                                              sigma_noise,
                                                              early_stopping=True,
                                                              initial_mode='load', equal_dictionaries=False,
                                                              dictionary_to_load=current_dictionary
                                                              )

            mse_omp_model.to(dtype=dtype, device=device)
            mse_omp_model.setRegularization(True, 5e-5)
            mse_omp_model.stopping_coef[:] = 1

            omp_model_mse_true_dict = models.Rand_Inner_Patch_OMP_Model(k_omp, round(math.sqrt(n)), round(math.sqrt(n)),
                                                                        m, sigma_noise,
                                                                        early_stopping=True,
                                                                        initial_mode='load', equal_dictionaries=True,
                                                                        dictionary_to_load=torch.from_numpy(true_D[i])
                                                                        )
            omp_model_mse_true_dict.to(dtype=dtype, device=device)
            omp_model_mse_true_dict.set_evaluation_mode(True)
            # omp_model_mse_true_dict.set_extra_input_flag(True)
            omp_model_mse_true_dict.stopping_coef[:] = 1

            k_sp = max_sparsity - 1
            sp_model = models.Inner_Patch_SP_Model(k_sp, round(math.sqrt(n)), round(math.sqrt(n)), m, sigma_noise,
                                                   early_stopping=False,
                                                   initial_mode='load', equal_dictionaries=False,
                                                   dictionary_to_load=current_dictionary
                                                   #  dictionary_to_load=torch.from_numpy(true_D[i])
                                                   )
            sp_model.to(dtype=dtype, device=device)
            sp_model.setRegularization(True, 5e-5)

            sp_model_true_dict = models.Inner_Patch_SP_Model(k_sp, round(math.sqrt(n)), round(math.sqrt(n)), m,
                                                             sigma_noise,
                                                             early_stopping=False,
                                                             initial_mode='load', equal_dictionaries=True,
                                                             dictionary_to_load=torch.from_numpy(true_D[i])
                                                             )
            sp_model_true_dict.to(dtype=dtype, device=device)
            sp_model_true_dict.evaluation_mode = True

            k_mp = k_omp
            mp_model = models.Patch_MP_Model(k_mp, round(math.sqrt(n)), round(math.sqrt(n)), m, sigma_noise,
                                             early_stopping=True,
                                             #  early_stopping=False,
                                             initial_mode='load',
                                             dictionary_to_load=current_dictionary,
                                             )

            mp_model.stopping_coef[:] = 1
            mp_model.setRegularization(True, 5e-5)
            mp_model.to(dtype=dtype, device=device)

            mp_model_true_dict = models.Patch_MP_Model(k_mp, round(math.sqrt(n)), round(math.sqrt(n)), m, sigma_noise,
                                                       early_stopping=True,
                                                       #  early_stopping=False,
                                                       initial_mode='load',
                                                       dictionary_to_load=torch.from_numpy(true_D[i]),
                                                       )

            mp_model_true_dict.stopping_coef[:] = 1
            mp_model_true_dict.setRegularization(True, 5e-5)
            mp_model_true_dict.to(dtype=dtype, device=device)

            k_lista = 7
            lista_model = models.LISTA_Patch_Model(k_lista, round(math.sqrt(n)), round(math.sqrt(n)), m, sigma_noise,
                                                   different_filter_threshold=True,
                                                   initial_mode='load', dictionary_to_load=current_dictionary
                                                   )
            lista_model.to(dtype=dtype, device=device)

            lista_model_true_dict = models.LISTA_Patch_Model(k_lista, round(math.sqrt(n)), round(math.sqrt(n)), m,
                                                             sigma_noise,
                                                             different_filter_threshold=True,
                                                             initial_mode='load',
                                                             dictionary_to_load=torch.from_numpy(true_D[i]))
            lista_model_true_dict.to(dtype=dtype, device=device)

            train_signals = data['train_signals'][i]
            train_representation = data['train_representation'][i]
            train_sparsity_map = data['train_sparsity_map'][i]
            train_transform = transforms.Compose(
                [ToTensor_Synthitic(device, dtype), AddWhiteGaussianNoise_Synthitic(sigma_noise)])
            # train_transform = transforms.Compose([ToTensor_Synthitic(device,dtype),AddWhiteGaussianNoise_Synthitic(sigma_noise),Calculate_Threshold()])
            train_dataset = Synthitic_Dataset(train_signals, train_representation, train_sparsity_map, true_D[i],
                                              transform=train_transform)
            train_snr = calculate_average_SNR(train_dataset)

            test_signals = data['test_signals'][i]
            test_representation = data['test_representation'][i]
            test_sparsity_map = data['test_sparsity_map'][i]
            test_transform = transforms.Compose(
                [ToTensor_Synthitic(device, dtype), AddWhiteGaussianNoise_Synthitic(sigma_noise)])
            # test_transform = transforms.Compose([ToTensor_Synthitic(device, dtype),AddWhiteGaussianNoise_Synthitic(sigma_noise),Calculate_Threshold()])
            test_dataset = Synthitic_Dataset(test_signals, test_representation, test_sparsity_map, true_D[i],
                                             transform=test_transform)
            test_snr = calculate_average_SNR(test_dataset)

            test_signals2 = data['test_signals2'][i]
            test_representation2 = data['test_representation2'][i]
            test_sparsity_map2 = data['test_sparsity_map2'][i]
            test_transform2 = transforms.Compose(
                [ToTensor_Synthitic(device, dtype), AddWhiteGaussianNoise_Synthitic(sigma_noise)])
            # test_transform2 = transforms.Compose([ToTensor_Synthitic(device, dtype),AddWhiteGaussianNoise_Synthitic(sigma_noise),Calculate_Threshold()])
            test_dataset2 = Synthitic_Dataset(test_signals2, test_representation2, test_sparsity_map2, true_D[i],
                                              transform=test_transform2)
            test_snr2 = calculate_average_SNR(test_dataset2)

            snr_list.append({'train_snr': train_snr, 'test_snr': test_snr, 'test_snr2': test_snr2})

            current_training_data = {}
            current_models_to_save = {}

            # omp_learning_rate = 0.001
            omp_learning_rate = 0.002
            mp_learning_rate = 0.002

            # sp_learning_rate = 0.001
            sp_learning_rate = 0.005
            # sp_learning_rate = 0.01

            omp_MSE_learning_rate = 0.01

            # lista_learning_rate = 0.0001
            lista_learning_rate = 0.00001
            omp_batch_size = 50
            lista_batch_size = 50
            sp_batch_size = 50

            omp_epochs = 300
            sp_epochs = 300
            mp_epochs = 300
            lista_epochs = 300
            omp_MSE_epochs = 100
            # omp_epochs = 3
            # sp_epochs = 3
            # mp_epochs = 3
            # lista_epochs = 3
            # omp_MSE_epochs = 3

            criterion = nn.MSELoss()

            MSE, sparsity = get_oracle_MSE(test_dataset2)
            print('Oracle average test loss = ', MSE)
            print('Oracle true dict average test sparsity = ', sparsity)
            oracle_mse_array[i, j] = MSE
            oracle_sparsity_array[i, j] = sparsity

            res = calculate_all_loss_model_synthitic_data(
                omp_model_true_dict, test_dataset2
                , True
            )

            print('OMP true dict average test loss = ', res['loss_average'])
            print('OMP true dict average test sparsity = ', res['sparsity_average'])
            omp_true_dict_mse_array[i, j] = res['loss_average']
            omp_true_dict_sparsity_array[i, j] = res['sparsity_average']

            res = calculate_all_loss_model_synthitic_data(
                omp_model_true_dict2, test_dataset2
                # ,True
            )

            print('OMP true dict2 average test loss = ', res['loss_average'])
            print('OMP true dict2 average test sparsity = ', res['sparsity_average'])
            omp_true_dict2_mse_array[i, j] = res['loss_average']
            omp_true_dict2_sparsity_array[i, j] = res['sparsity_average']

            res = calculate_all_loss_model_synthitic_data(
                omp_model_mse_true_dict, test_dataset2
                # ,True
            )

            print('OMP MSE true dict average test loss = ', res['loss_average'])
            print('OMP MSE true dict average test sparsity = ', res['sparsity_average'])
            omp_true_dict_MMSE_mse_array[i, j] = res['loss_average']
            omp_true_dict_MMSE_sparsity_array[i, j] = res['sparsity_average']

            if (MMSE_run_list[j]):
                res = calculate_all_loss_model_synthitic_data(
                    sp_model_true_dict, test_dataset2
                    # ,True
                )

                print('SP true dict average test loss = ', res['loss_average'])
                print('SP true dict average test sparsity = ', res['sparsity_average'])
                sp_true_dict_mse_array[i, j] = res['loss_average']
                sp_true_dict_sparsity_array[i, j] = res['sparsity_average']

                res = calculate_all_loss_model_synthitic_data(
                    mp_model_true_dict, test_dataset2
                    # ,True
                )

                print('MP true dict average test loss = ', res['loss_average'])
                print('MP true dict average test sparsity = ', res['sparsity_average'])
                mp_true_dict_mse_array[i, j] = res['loss_average']
                mp_true_dict_sparsity_array[i, j] = res['sparsity_average']

                tmp_trainig_data = train_model_synthitic_data(train_dataset, test_dataset, mse_omp_model,
                                                              'MSE_OMP_model',
                                                              criterion, batch_size=omp_batch_size,
                                                              epochs=omp_MSE_epochs,
                                                              learning_rate=omp_MSE_learning_rate, optim_type='Adam'
                                                              #  ,extra_input_mode=True
                                                              , extra_info_function=lambda x_model: get_model_dict_info(
                        torch.from_numpy(true_D[i]), x_model)
                                                              )
                current_training_data['MSE_OMP_model'] = tmp_trainig_data
                current_models_to_save['MSE_OMP_model'] = mse_omp_model.state_dict()
                # omp_model.tune_sigma_coef(test_dataset,calculate_all_loss_model_synthitic_data,True,torch.arange(-0.5,0.5,0.01))
                # tmp_omp_loss_array , tmp1 = train_model_synthitic_data(train_dataset, test_dataset, omp_model, 'OMP_model',
                #                                                        criterion, batch_size=omp_batch_size, epochs=omp_epochs,
                #                            learning_rate=omp_learning_rate, optim_type='Adam')

                res = calculate_all_loss_model_synthitic_data(mse_omp_model, test_dataset2
                                                              # ,True
                                                              )

                print('OMP MSE model average test loss = ', res['loss_average'])
                print('OMP MSE model average test sparsity = ', res['sparsity_average'])
                omp_MMSE_mse_array[i, j] = res['loss_average']
                omp_MMSE_sparsity_array[i, j] = res['sparsity_average']

                tmp_trainig_data = train_model_synthitic_data(train_dataset, test_dataset, sp_model, 'SP_model',
                                                              criterion, batch_size=omp_batch_size, epochs=sp_epochs,
                                                              learning_rate=sp_learning_rate, optim_type='Adam'
                                                              #  ,extra_input_mode=True
                                                              #  ,enable_schedluer=True
                                                              , extra_info_function=lambda x_model: get_model_dict_info(
                        torch.from_numpy(true_D[i]), x_model)
                                                              )

                current_training_data['SP_model'] = tmp_trainig_data
                current_models_to_save['SP_model'] = sp_model.state_dict()

                res = calculate_all_loss_model_synthitic_data(sp_model, test_dataset2
                                                              # ,True
                                                              )

                print('SP average test loss = ', res['loss_average'])
                print('SP average test sparsity = ', res['sparsity_average'])
                sp_mse_array[i, j] = res['loss_average']
                sp_sparsity_array[i, j] = res['sparsity_average']

                tmp_trainig_data = train_model_synthitic_data(train_dataset, test_dataset, mp_model, 'MP_model',
                                                              criterion, batch_size=omp_batch_size, epochs=mp_epochs,
                                                              learning_rate=mp_learning_rate, optim_type='Adam'
                                                              #  ,extra_input_mode=True
                                                              #  ,enable_schedluer=True
                                                              , extra_info_function=lambda x_model: get_model_dict_info(
                        torch.from_numpy(true_D[i]), x_model)
                                                              )

                current_training_data['MP_model'] = tmp_trainig_data
                current_models_to_save['MP_model'] = mp_model.state_dict()

                res = calculate_all_loss_model_synthitic_data(mp_model, test_dataset2
                                                              # ,True
                                                              )

                print('MP average test loss = ', res['loss_average'])
                print('MP average test sparsity = ', res['sparsity_average'])
                mp_mse_array[i, j] = res['loss_average']
                mp_sparsity_array[i, j] = res['sparsity_average']

            tmp_trainig_data = train_model_synthitic_data(train_dataset, test_dataset, omp_model, 'OMP_model',
                                                          criterion, batch_size=omp_batch_size, epochs=omp_epochs,
                                                          learning_rate=omp_learning_rate, optim_type='Adam'
                                                          #  ,extra_input_mode=True
                                                          #  ,enable_schedluer=True
                                                          , extra_info_function=lambda x_model: get_model_dict_info(
                    torch.from_numpy(true_D[i]), x_model)
                                                          )

            current_training_data['OMP_model'] = tmp_trainig_data
            current_models_to_save['OMP_model'] = omp_model.state_dict()

            res = calculate_all_loss_model_synthitic_data(omp_model, test_dataset2
                                                          # ,True
                                                          )

            print('OMP average test loss = ', res['loss_average'])
            print('OMP average test sparsity = ', res['sparsity_average'])
            omp_mse_array[i, j] = res['loss_average']
            omp_sparsity_array[i, j] = res['sparsity_average']
            true_sparsity_array[i, j] = res['true_sparsity']

            tmp_trainig_data = train_model_synthitic_data(train_dataset, test_dataset, omp_model2,
                                                          'OMP_model_true_cardinality',
                                                          criterion, batch_size=omp_batch_size, epochs=omp_epochs,
                                                          learning_rate=omp_learning_rate, optim_type='Adam'
                                                          , extra_input_mode=True
                                                          #  ,enable_schedluer=True
                                                          , extra_info_function=lambda x_model: get_model_dict_info(
                    torch.from_numpy(true_D[i]), x_model)
                                                          )

            current_training_data['OMP_model_true_cardinality'] = tmp_trainig_data
            current_models_to_save['OMP_model_true_cardinality'] = omp_model2.state_dict()

            res = calculate_all_loss_model_synthitic_data(omp_model2, test_dataset2
                                                          , True
                                                          )

            print('OMP true cardinality average test loss = ', res['loss_average'])
            print('OMP true cardinality average test sparsity = ', res['sparsity_average'])
            omp_mse_true_cardinality_array[i, j] = res['loss_average']
            omp_sparsity_true_cardinality_array[i, j] = res['sparsity_average']

            if (MMSE_run_list[j]):
                post_training_mse_omp_model = models.Rand_Inner_Patch_OMP_Model(k_omp, round(math.sqrt(n)),
                                                                                round(math.sqrt(n)), m, sigma_noise,
                                                                                early_stopping=True,
                                                                                #  early_stopping=False,
                                                                                initial_mode='load',
                                                                                equal_dictionaries=False,
                                                                                dictionary_to_load=current_dictionary,
                                                                                )

                post_training_mse_omp_model.to(dtype=dtype, device=device)
                # post_training_mse_omp_model.set_extra_input_flag(True)
                post_training_mse_omp_model.load_state_dict(omp_model.state_dict())
                res = calculate_all_loss_model_synthitic_data(
                    post_training_mse_omp_model, test_dataset2
                    # ,True
                )

                print('OMP post training MSE average test loss = ', res['loss_average'])
                print('OMP post training MSE average test sparsity = ', res['sparsity_average'])
                omp_post_training_MMSE_mse_array[i, j] = res['loss_average']
                omp_post_training_MMSE_sparsity_array[i, j] = res['sparsity_average']

            tmp_trainig_data = train_model_synthitic_data(train_dataset, test_dataset, lista_model, 'LISTA_model',
                                                          criterion, batch_size=lista_batch_size,
                                                          epochs=lista_epochs,
                                                          learning_rate=lista_learning_rate, optim_type='Adam'
                                                          #  ,enable_schedluer=True
                                                          , extra_info_function=lambda x_model: get_model_dict_info(
                    torch.from_numpy(true_D[i]), x_model)
                                                          )
            current_training_data['LISTA_model'] = tmp_trainig_data
            current_models_to_save['LISTA_model'] = lista_model.state_dict()

            res = calculate_all_loss_model_synthitic_data(lista_model, test_dataset2)

            print('LISTA average test loss = ', res['loss_average'])
            print('LISTA average test sparsity = ', res['sparsity_average'])
            lista_mse_array[i, j] = res['loss_average']
            lista_sparsity_array[i, j] = res['sparsity_average']

            res = calculate_all_loss_model_synthitic_data(
                lista_model_true_dict,
                test_dataset2)

            print('LISTA true dict average test loss = ', res['loss_average'])
            print('LISTA true dict average test sparsity = ', res['sparsity_average'])
            lista_true_dict_mse_array[i, j] = res['loss_average']
            lista_true_dict_sparsity_array[i, j] = res['sparsity_average']

            training_procedure_data.append(current_training_data)
            models_to_save.append(current_models_to_save)

            omp_training_loss = [val['loss_average'] for val in current_training_data['OMP_model']]
            lista_training_loss = [val['loss_average'] for val in current_training_data['LISTA_model']]

            import matplotlib.pyplot as plt
            plt.plot(range(len(omp_training_loss)), omp_training_loss, 'b-', label='LGM')
            # plt.plot(range(sp_epochs + 1), tmp_sp_loss_array.clone().detach().cpu().numpy(), 'g^-', label='SP')
            plt.plot(range(len(lista_training_loss)), lista_training_loss, 'r-', label='LISTA')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.legend()
            plt.show()

    # assert (torch.dist(omp_true_sparsity_array,lista_true_sparsity_array)<1e-5)
    dict_to_save = {}
    dict_to_save['true_data'] = data
    dict_to_save['initial_training_dictionary'] = D

    dict_to_save['omp_mse_array'] = omp_mse_array
    dict_to_save['omp_sparsity_array'] = omp_sparsity_array
    dict_to_save['sp_mse_array'] = sp_mse_array
    dict_to_save['sp_sparsity_array'] = sp_sparsity_array
    dict_to_save['mp_mse_array'] = mp_mse_array
    dict_to_save['mp_sparsity_array'] = mp_sparsity_array
    dict_to_save['lista_mse_array'] = lista_mse_array
    dict_to_save['lista_sparsity_array'] = lista_sparsity_array
    dict_to_save['true_sparsity_array'] = true_sparsity_array

    dict_to_save['oracle_mse_array'] = oracle_mse_array
    dict_to_save['oracle_sparsity_array'] = oracle_sparsity_array
    dict_to_save['omp_true_dict_mse_array'] = omp_true_dict_mse_array
    dict_to_save['omp_true_dict_sparsity_array'] = omp_true_dict_sparsity_array
    dict_to_save['omp_true_dict2_mse_array'] = omp_true_dict2_mse_array
    dict_to_save['omp_true_dict2_sparsity_array'] = omp_true_dict2_sparsity_array
    dict_to_save['lista_true_dict_mse_array'] = lista_true_dict_mse_array
    dict_to_save['lista_true_dict_sparsity_array'] = lista_true_dict_sparsity_array

    dict_to_save['mp_true_dict_mse_array'] = mp_true_dict_mse_array
    dict_to_save['mp_true_dict_sparsity_array'] = mp_true_dict_sparsity_array
    dict_to_save['sp_true_dict_mse_array'] = sp_true_dict_mse_array
    dict_to_save['sp_true_dict_sparsity_array'] = sp_true_dict_sparsity_array

    dict_to_save['omp_MMSE_mse_array'] = omp_MMSE_mse_array
    dict_to_save['omp_MMSE_sparsity_array'] = omp_MMSE_sparsity_array

    dict_to_save['omp_post_training_MMSE_mse_array'] = omp_post_training_MMSE_mse_array
    dict_to_save['omp_post_training_MMSE_sparsity_array'] = omp_post_training_MMSE_sparsity_array

    dict_to_save['omp_true_dict_MMSE_mse_array'] = omp_true_dict_MMSE_mse_array
    dict_to_save['omp_true_dict_MMSE_sparsity_array'] = omp_true_dict_MMSE_sparsity_array

    dict_to_save['omp_mse_true_cardinality_array'] = omp_mse_true_cardinality_array
    dict_to_save['omp_sparsity_true_cardinality_array'] = omp_sparsity_true_cardinality_array

    dict_to_save['snr_list'] = snr_list
    dict_to_save['training_procedure_data'] = training_procedure_data
    dict_to_save['models_to_save'] = models_to_save
    dict_to_save['MMSE_run_list'] = MMSE_run_list
    dict_to_save['sigma_noise_range'] = sigma_noise_range

    # specify save location
    # save_location = 'data.pt'
    save_location = '/content/drive/My Drive/syn1/data.pt'
    torch.save(dict_to_save, save_location)
    # drive.flush_and_unmount()
    print('saved successfully')


np.random.seed(1995)

# dtype = torch.float64
dtype = torch.float32
# device = torch.device("cpu" )# Uncomment this to run on CPU
device = torch.device("cuda")  # Uncomment this to run on GPU
run_experiemnts()


# part 2:loading and presenting the results


import torch
import numpy as np


# specify load locations by noise order
data_lst = [
            torch.load('/content/drive/My Drive/syn1/data.pt',map_location = torch.device('cpu'))
            ,torch.load('/content/drive/My Drive/syn2/data.pt',map_location = torch.device('cpu'))
            ,torch.load('/content/drive/My Drive/syn3/data.pt',map_location = torch.device('cpu'))
            ,torch.load('/content/drive/My Drive/syn4/data.pt',map_location = torch.device('cpu'))
            ,torch.load('/content/drive/My Drive/syn5/data.pt',map_location = torch.device('cpu'))
            ,torch.load('/content/drive/My Drive/syn6/data.pt',map_location = torch.device('cpu'))
            ]


data1 = data_lst[0]['true_data']


# making sure that they have the same data
tmp_lst = [i['true_data'] for i in data_lst]
for data2 in tmp_lst:
    for key in data1.keys():
        if(type(data1[key])==type(5)):
            continue
        diff = np.linalg.norm(data1[key]-data2[key])
        print(data1[key].shape)
        assert(diff < 1e-5)


dict_to_load = {}
key_list1 = ['omp_mse_array','omp_sparsity_array','sp_mse_array','sp_sparsity_array','mp_mse_array','mp_sparsity_array',
             'lista_mse_array','lista_sparsity_array','true_sparsity_array','oracle_mse_array','oracle_sparsity_array',
              'omp_true_dict_mse_array','omp_true_dict_sparsity_array','omp_true_dict2_mse_array','omp_true_dict2_sparsity_array',
             'lista_true_dict_mse_array','lista_true_dict_sparsity_array','mp_true_dict_mse_array','mp_true_dict_sparsity_array',
              'sp_true_dict_mse_array','sp_true_dict_sparsity_array','omp_MMSE_mse_array','omp_MMSE_sparsity_array',
             'omp_post_training_MMSE_mse_array','omp_post_training_MMSE_sparsity_array','omp_true_dict_MMSE_mse_array',
              'omp_true_dict_MMSE_sparsity_array','omp_mse_true_cardinality_array','omp_sparsity_true_cardinality_array'
             ]

key_list2 = ['sigma_noise_range','snr_list','training_procedure_data']

for key in key_list1:
    dict_to_load[key] = torch.cat(tuple([i[key] for i in data_lst]),dim=1)

for key in key_list2:
    lst = [i[key][0] for i in data_lst]
    # print(lst)
    dict_to_load[key] = lst



omp_mse_array = dict_to_load['omp_mse_array'].mean(dim=0).numpy()
omp_sparsity_array = dict_to_load['omp_sparsity_array'].mean(dim=0).numpy()
sp_mse_array = dict_to_load['sp_mse_array'].mean(dim=0).numpy()
sp_sparsity_array = dict_to_load['sp_sparsity_array'].mean(dim=0).numpy()
mp_mse_array = dict_to_load['mp_mse_array'].mean(dim=0).numpy()
mp_sparsity_array = dict_to_load['mp_sparsity_array'].mean(dim=0).numpy()

lista_mse_array = dict_to_load['lista_mse_array'].mean(dim=0).numpy()
lista_sparsity_array = dict_to_load['lista_sparsity_array'].mean(dim=0).numpy()
true_sparsity_array = dict_to_load['true_sparsity_array'].mean(dim=0).numpy()
sigma_noise_range = dict_to_load['sigma_noise_range']

oracle_mse_array = dict_to_load['oracle_mse_array'].mean(dim=0).numpy()
oracle_sparsity_array = dict_to_load['oracle_sparsity_array'].mean(dim=0).numpy()
omp_true_dict_mse_array = dict_to_load['omp_true_dict_mse_array'].mean(dim=0).numpy()
omp_true_dict_sparsity_array = dict_to_load['omp_true_dict_sparsity_array'].mean(dim=0).numpy()
omp_true_dict2_mse_array = dict_to_load['omp_true_dict2_mse_array'].mean(dim=0).numpy()
omp_true_dict2_sparsity_array = dict_to_load['omp_true_dict2_sparsity_array'].mean(dim=0).numpy()
lista_true_dict_mse_array = dict_to_load['lista_true_dict_mse_array'].mean(dim=0).numpy()
lista_true_dict_sparsity_array = dict_to_load['lista_true_dict_sparsity_array'].mean(dim=0).numpy()


sp_true_dict_mse_array = dict_to_load['sp_true_dict_mse_array'].mean(dim=0).numpy()
sp_true_dict_sparsity_array = dict_to_load['sp_true_dict_sparsity_array'].mean(dim=0).numpy()
mp_true_dict_mse_array = dict_to_load['mp_true_dict_mse_array'].mean(dim=0).numpy()
mp_true_dict_sparsity_array = dict_to_load['mp_true_dict_sparsity_array'].mean(dim=0).numpy()

omp_MMSE_mse_array = dict_to_load['omp_MMSE_mse_array'].mean(dim=0).numpy()
omp_MMSE_sparsity_array = dict_to_load['omp_MMSE_sparsity_array'].mean(dim=0).numpy()

omp_post_training_MMSE_mse_array = dict_to_load['omp_post_training_MMSE_mse_array'].mean(dim=0).numpy()
omp_post_training_MMSE_sparsity_array = dict_to_load['omp_post_training_MMSE_sparsity_array'].mean(dim=0).numpy()

omp_true_dict_MMSE_mse_array = dict_to_load['omp_true_dict_MMSE_mse_array'].mean(dim=0).numpy()
omp_true_dict_MMSE_sparsity_array = dict_to_load['omp_true_dict_MMSE_sparsity_array'].mean(dim=0).numpy()

omp_mse_true_cardinality_array = dict_to_load['omp_mse_true_cardinality_array'].mean(dim=0).numpy()
omp_sparsity_true_cardinality_array = dict_to_load['omp_sparsity_true_cardinality_array'].mean(dim=0).numpy()

print('SNR list = ',dict_to_load['snr_list'])
# print('MMSE_run_list = ',dict_to_load['MMSE_run_list'])


noise_index = 5
# noise_index = 0

LGM_D_array = [val['extra_info']['D_L_w_e_dist'] for val in dict_to_load['training_procedure_data'][noise_index]['OMP_model']]
LGM_D2_array = [val['extra_info']['D_L_d_dist'] for val in dict_to_load['training_procedure_data'][noise_index]['OMP_model']]

LGM_MSE_D_array = [val['extra_info']['D_L_w_e_dist'] for val in dict_to_load['training_procedure_data'][noise_index]['MSE_OMP_model']]
LGM_MSE_D2_array = [val['extra_info']['D_L_d_dist'] for val in dict_to_load['training_procedure_data'][noise_index]['MSE_OMP_model']]

LSP_D_array = [val['extra_info']['D_L_w_e_dist'] for val in dict_to_load['training_procedure_data'][noise_index]['SP_model']]
LSP_D2_array = [val['extra_info']['D_L_d_dist'] for val in dict_to_load['training_procedure_data'][noise_index]['SP_model']]

LMP_D_array = [val['extra_info']['D_L_w_e_dist'] for val in dict_to_load['training_procedure_data'][noise_index]['MP_model']]
LMP_D2_array = [val['extra_info']['D_L_d_dist'] for val in dict_to_load['training_procedure_data'][noise_index]['MP_model']]

LISTA_D_L_w_e_array = [val['extra_info']['D_L_w_e_dist'] for val in dict_to_load['training_procedure_data'][noise_index]['LISTA_model']]
LISTA_D_L_w_d_array = [val['extra_info']['D_L_w_d_dist'] for val in dict_to_load['training_procedure_data'][noise_index]['LISTA_model']]
LISTA_D_L_d_array = [val['extra_info']['D_L_d_dist'] for val in dict_to_load['training_procedure_data'][noise_index]['LISTA_model']]

LGM_training_sparsity_array = [val['sparsity_average'] for val in dict_to_load['training_procedure_data'][noise_index]['OMP_model']]
LGM_MSE_training_sparsity_array = [val['sparsity_average'] for val in dict_to_load['training_procedure_data'][noise_index]['MSE_OMP_model']]
LSP_training_sparsity_array = [val['sparsity_average'] for val in dict_to_load['training_procedure_data'][noise_index]['SP_model']]
LMP_training_sparsity_array = [val['sparsity_average'] for val in dict_to_load['training_procedure_data'][noise_index]['MP_model']]
LISTA_training_sparsity_array = [val['sparsity_average'] for val in dict_to_load['training_procedure_data'][noise_index]['LISTA_model']]



extended_sigma_noise_range = [0.9*sigma_noise_range[0]]+sigma_noise_range+[1.05*sigma_noise_range[-1]]


import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

plt.plot(sigma_noise_range,omp_mse_array,'bo-',label = 'LGM')
# plt.plot(sigma_noise_range,omp_post_training_MMSE_mse_array,'b+-',label = 'LGM Post-training MSE')
# plt.plot(sigma_noise_range,omp_MMSE_mse_array,'gH-',label = 'LGM MSE')
plt.plot(sigma_noise_range,omp_mse_true_cardinality_array,'yH-',label = 'LGM True Cardinality')
# plt.plot(true_sparsity_array, sp_mse_array, 'g^-', label='SP')
plt.plot(sigma_noise_range, lista_mse_array,'r*-',label = 'LISTA')

plt.plot(sigma_noise_range, oracle_mse_array, 'cD-', label='Oracle')
plt.plot(sigma_noise_range, omp_true_dict_mse_array, 'mx-', label='OMP True Dict & Cardinality')
plt.plot(sigma_noise_range, omp_true_dict2_mse_array, 'P-',color='chocolate', label='OMP True Dict')
# plt.plot(sigma_noise_range, omp_true_dict_MMSE_mse_array, 'ks-', label='OMP True Dict MSE')
# plt.plot(sigma_noise_range, lista_true_dict_mse_array, 'yv-', label='ISTA True Dict')
plt.xlabel('Noise Level')
plt.ylabel('MSE')
plt.legend()
plt.grid()
plt.minorticks_on()
plt.savefig('MSE.png', dpi=400)
plt.show()

plt.plot(sigma_noise_range,omp_mse_array,'bo-',label = 'LGM')
plt.plot(sigma_noise_range,omp_post_training_MMSE_mse_array,'b+-',label = 'LGM Post-training MMSE')
plt.plot(sigma_noise_range,omp_MMSE_mse_array,'gH-',label = 'LGM MMSE')
plt.plot(sigma_noise_range, lista_mse_array,'r*-',label = 'LISTA')

plt.plot(sigma_noise_range, oracle_mse_array, 'cD-', label='Oracle')
plt.plot(sigma_noise_range, omp_true_dict_MMSE_mse_array, 'ks-', label='OMP True Dict MMSE')
plt.xlabel('Noise Level')
plt.ylabel('MSE')
plt.legend()
plt.grid()
plt.minorticks_on()
plt.savefig('MSE2.png', dpi=400)
plt.show()

# plt.plot(sigma_noise_range,omp_mse_array,'bo-',label = 'LGM')
# plt.plot(sigma_noise_range,omp_mse_true_cardinality_array,'yH-',label = 'LGM True Cardinality')
# plt.plot(sigma_noise_range,sp_mse_array,'>-',color='gray',label = 'L-SP')
# plt.plot(sigma_noise_range,mp_mse_array,'3-',color='peru',label = 'L-MP')
# plt.plot(sigma_noise_range,mp_true_dict_mse_array,'4-',color='olive',label = 'MP True Dict')
# # plt.plot(sigma_noise_range, lista_mse_array,'r*-',label = 'LISTA')
#
# plt.plot(sigma_noise_range, oracle_mse_array, 'cD-', label='Oracle')
# plt.plot(sigma_noise_range, omp_true_dict_mse_array, 'mx-', label='OMP True Dict & Cardinality')
# plt.plot(sigma_noise_range, omp_true_dict2_mse_array, 'P-',color='chocolate', label='OMP True Dict')
#
# plt.xlabel('Noise Level')
# plt.ylabel('MSE')
# plt.legend()
# plt.grid()
# plt.minorticks_on()
# plt.savefig('MSE3.png', dpi=400)
# plt.show()




plt.plot(sigma_noise_range, omp_sparsity_array, 'bo-', label='LGM')
# plt.plot(sigma_noise_range,omp_post_training_MMSE_sparsity_array,'b+-',label = 'LGM Post-training MSE')
# plt.plot(sigma_noise_range,omp_MMSE_sparsity_array,'gH-',label = 'LGM MSE')
plt.plot(sigma_noise_range,omp_sparsity_true_cardinality_array,'yH-',label = 'LGM True Cardinality')
# plt.plot(true_sparsity_array, sp_sparsity_array, 'g^-', label='SP')
plt.plot(sigma_noise_range, lista_sparsity_array, 'r*-', label='LISTA')

plt.plot(sigma_noise_range, oracle_sparsity_array, 'cD-', label='Oracle')
plt.plot(sigma_noise_range, omp_true_dict_sparsity_array, 'mx-', label='OMP True Dict & Cardinality')
plt.plot(sigma_noise_range, omp_true_dict2_sparsity_array, 'P-',color='chocolate', label='OMP True Dict')
# plt.plot(sigma_noise_range, omp_true_dict_MMSE_sparsity_array, 'ks-', label='OMP True Dict MSE')
# plt.plot(sigma_noise_range, lista_true_dict_sparsity_array, 'yv-', label='ISTA True Dict')
plt.plot(extended_sigma_noise_range, [30]*len(extended_sigma_noise_range), 'k--')
plt.plot(extended_sigma_noise_range, [-15]*len(extended_sigma_noise_range), 'k--')
plt.plot([extended_sigma_noise_range[0]]*2, [-15,30], 'k--')
plt.plot([extended_sigma_noise_range[-1]]*2, [-15,30], 'k--')
plt.xlabel('Noise Level')
plt.ylabel('Restored Signal Cardinality')
plt.legend()
plt.grid()
plt.savefig('Sparsity.png', dpi=400)
plt.show()

plt.plot(sigma_noise_range, omp_sparsity_array, 'bo-', label='LGM')
# plt.plot(sigma_noise_range,omp_post_training_MMSE_sparsity_array,'b+-',label = 'LGM Post-training MSE')
# plt.plot(sigma_noise_range,omp_MMSE_sparsity_array,'gH-',label = 'LGM MSE')
plt.plot(sigma_noise_range,omp_sparsity_true_cardinality_array,'yH-',label = 'LGM True Cardinality')
# plt.plot(true_sparsity_array, sp_sparsity_array, 'g^-', label='SP')
# plt.plot(sigma_noise_range, lista_sparsity_array, 'r*-', label='LISTA')

plt.plot(sigma_noise_range, oracle_sparsity_array, 'cD-', label='Oracle')
plt.plot(sigma_noise_range, omp_true_dict_sparsity_array, 'mx-', label='OMP True Dict & Cardinality')
plt.plot(sigma_noise_range, omp_true_dict2_sparsity_array, 'P-',color='chocolate', label='OMP True Dict')
# plt.plot(sigma_noise_range, omp_true_dict_MMSE_sparsity_array, 'ks-', label='OMP True Dict MSE')
# plt.plot(sigma_noise_range, lista_true_dict_sparsity_array, 'yv-', label='ISTA True Dict')
plt.xlabel('Noise Level')
plt.ylabel('Restored Signal Cardinality')
plt.legend()
plt.grid()

plt.savefig('Sparsity2.png', dpi=400)
plt.show()

# plt.plot(sigma_noise_range,omp_sparsity_array,'bo-',label = 'LGM')
# plt.plot(sigma_noise_range,omp_sparsity_true_cardinality_array,'yH-',label = 'LGM True Cardinality')
# plt.plot(sigma_noise_range,sp_sparsity_array,'>-',color='gray',label = 'L-SP')
# plt.plot(sigma_noise_range,mp_sparsity_array,'3-',color='peru',label = 'L-MP')
# plt.plot(sigma_noise_range,mp_true_dict_sparsity_array,'4-',color='olive',label = 'MP True Dict')
# # plt.plot(sigma_noise_range, lista_mse_array,'r*-',label = 'LISTA')
#
# plt.plot(sigma_noise_range, oracle_sparsity_array, 'cD-', label='Oracle')
# plt.plot(sigma_noise_range, omp_true_dict_sparsity_array, 'mx-', label='OMP True Dict & Cardinality')
# plt.plot(sigma_noise_range, omp_true_dict2_sparsity_array, 'P-',color='chocolate', label='OMP True Dict')
#
# plt.xlabel('Noise Level')
# plt.ylabel('Restored Signal Cardinality')
# plt.legend()
# plt.grid()
# plt.savefig('Sparsity3.png', dpi=400)
# plt.show()


plt.plot(range(len(LGM_D_array)),LGM_D_array,'b-',label = 'LGM D')
plt.plot(range(len(LGM_D2_array)),LGM_D2_array,'b--',label = 'LGM D2')


plt.plot(range(len(LGM_MSE_D_array)),LGM_MSE_D_array,'y-',label = 'LGM MMSE D')
plt.plot(range(len(LGM_MSE_D2_array)),LGM_MSE_D2_array,'y--',label = 'LGM MMSE D2')

# plt.plot(range(len(LSP_D_array)),LSP_D_array,'-',color='gray',label = 'L-SP D')
# plt.plot(range(len(LSP_D2_array)),LSP_D2_array,'--',color='gray',label = 'L-SP D2')
#
# plt.plot(range(len(LMP_D_array)),LMP_D_array,'-',color='peru',label = 'L-MP D')
# plt.plot(range(len(LMP_D2_array)),LMP_D2_array,'--',color='peru',label = 'L-MP D2')

plt.plot(range(len(LISTA_D_L_w_e_array)), LISTA_D_L_w_e_array,'r-',label = 'LISTA W')
plt.plot(range(len(LISTA_D_L_w_d_array)), LISTA_D_L_w_d_array,'r--',label = 'LISTA D1')
plt.plot(range(len(LISTA_D_L_d_array)), LISTA_D_L_d_array,'r:',label = 'LISTA D2')

plt.xlabel('Epoch')
plt.ylabel('Distance From The True Dictionary')
plt.legend()
plt.grid()
plt.minorticks_on()
plt.savefig('D_dist.png', dpi=600)
plt.show()






plt.plot(range(len(LGM_training_sparsity_array)),LGM_training_sparsity_array,'b-',label = 'LGM')
plt.plot(range(len(LGM_MSE_training_sparsity_array)),LGM_MSE_training_sparsity_array,'y-',label = 'LGM MMSE')
plt.plot(range(len(LSP_training_sparsity_array)),LSP_training_sparsity_array,'-',color='gray',label = 'L-SP')
plt.plot(range(len(LMP_training_sparsity_array)),LMP_training_sparsity_array,'-',color='peru',label = 'L-MP')
# plt.plot(range(len(LISTA_training_sparsity_array)), LISTA_training_sparsity_array,'r-',label = 'LISTA')

plt.xlabel('Epoch')
plt.ylabel('Restored Signal Sparsity')
plt.legend()
plt.grid()
plt.minorticks_on()
plt.savefig('Training_Sparsity.png', dpi=400)
plt.show()