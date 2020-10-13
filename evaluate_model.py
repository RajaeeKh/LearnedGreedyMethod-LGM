import Models as aux
from Models import Log_Loss_2
from Models import Tmp_Loss
import torch
from torchvision import transforms, utils


parameters_config = {}
parameters_config['denoising'] = {}

parameters_config['denoising']['basic'] = {}
parameters_config['denoising']['advanced'] = {}

parameters_config['denoising']['basic'] ={
    'model_parameters':{
        'k':10
        ,'n_row':8
        ,'n_col':8
        ,'m':4*8*8
        ,'initial_mode':'DCT'
        ,'early_stopping':False
        ,'learned_weighted_averaging':False
        ,'add_DC_atom':True
        ,'apply_attention_mode':True
    }
    ,'class':aux.OMP_Model
    ,'colored':False
    ,'rain_dataset':False
    , 'get_extra_info_flag': False

}
parameters_config['denoising']['advanced'] = {
    'model_parameters':{
            'initial_mode':'DCT'
            ,'early_stopping':False
            ,'learned_weighted_averaging':False
            ,'add_DC_atom':True
            ,'apply_attention_mode':True
    }
    ,'class':aux.OMP_Model_Global_MultiScale
    ,'colored':False
    ,'rain_dataset':False
    , 'get_extra_info_flag': False
}

parameters_config['deraining'] = {
    'model_parameters':{
        'k':20
        ,'n_row':8
        ,'n_col':8
        ,'m':9*8*8
        ,'initial_mode':'random'
        ,'early_stopping':True
        ,'learned_weighted_averaging':False
        ,'add_DC_atom':True
        ,'channels':3
        ,'enable_rain_net':True
    }
    ,'class':aux.OMP_Model
    ,'colored':True
    ,'rain_dataset':True
    ,'get_extra_info_flag':True

}

# choose config
application = 'denoising'
# application = 'deraining'
print(application)

# dtype = torch.float64
dtype = torch.float32
# device = torch.device("cpu") # Uncomment this to run on CPU
device = torch.device("cuda") # Uncomment this to run on GPU

if(application=='denoising'):
    testset_location=['BSD68/']
    testset2_location=['Set12/']

    # true_sigma_noise = 15
    true_sigma_noise = 25
    # true_sigma_noise = 50

    sigma_noise = true_sigma_noise
    print(sigma_noise)
    sigma_noise = float(sigma_noise)/255

    # choose config
    mode = 'basic'
    # mode = 'advanced'

    config = parameters_config[application][mode]

    test_composed = transforms.Compose([aux.AddWhiteGaussianNoise(sigma_noise),
                                        aux.ToTensor(device, dtype), aux.Normlize()])

    print(mode)

    # load_loaction = 'pre_trained_models/denoising/' + mode + '_' + str(true_sigma_noise) + '/LGM_model.pt'
    load_loaction = 'pre_trained_models/denoising/' + mode + '_' + str(true_sigma_noise) + '/top_LGM_model.pt'
    get_extra_info_flag = False

if(application=='deraining'):
    testset_location = ['RainData/Rain100L/']
    testset2_location = ['RainData/test12/']


    sigma_noise = 4
    sigma_noise = float(sigma_noise) / 255

    config = parameters_config[application]

    test_composed = transforms.Compose([aux.ToTensor(device, dtype),
         lambda sample: dict(list(sample.items()) + [('mean', 0)])])

    load_loaction = 'pre_trained_models/deraining/LGM_model.pt'
    get_extra_info_flag = True





config['model_parameters']['sigma_noise']=sigma_noise
print(application)
print('parameters config:')
print(config)






test_set = aux.ImageDataSet(testset_location, transform=test_composed
                            ,colored=config['colored']
                            ,rain_dataset=config['rain_dataset']
                            )
print('test_set sample=',len(test_set))
test_set2 = aux.ImageDataSet(testset2_location,transform=test_composed
                            ,colored=config['colored']
                            ,rain_dataset=config['rain_dataset']
                             )
print('test_set2 sample=',len(test_set2))

model = config['class'](**config['model_parameters'])
model.to(dtype = dtype,device=device)




checkpoint = torch.load(load_loaction)
model_data = checkpoint['model_state_dict']
model.load_state_dict(model_data)
model.set_enable_cholesky_decomposition_flag(True)
model.get_extra_info_flag(get_extra_info_flag)


testset_save_location1 ='res1/'
test_loss, test_local_sparsity, test_psnr = aux.calculate_all_loss_model(
                    model, test_set,enable_extra_input=False
                    ,save_location = testset_save_location1
)
print(len(test_set))
print(test_psnr)

testset_save_location2 ='res2/'
test_loss, test_local_sparsity, test_psnr = aux.calculate_all_loss_model(
                    model, test_set2,enable_extra_input=False
                    ,save_location = testset_save_location2
)

print(len(test_set2))
print(test_psnr)