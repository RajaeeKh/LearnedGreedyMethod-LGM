import Models as aux
import torch
from torchvision import transforms, utils




parameters_config = {}
parameters_config['denoising'] = {}

parameters_config['denoising']['basic'] = {}
parameters_config['denoising']['advanced'] = {}

parameters_config['denoising']['basic'] ={
    'crop_size':(100,100)
    ,'train_set_settings':{
        'data':['CBSD432/']  #TODO: Fix
        ,'read_from_hardisk':False
    }
    ,'batch_size':8
    ,'epochs':3000
    ,'learning_rate':0.002
    ,'enable_scheduler':False
    ,'evaluation_cycle': (20,10)
    ,'model_parameters':{
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
    ,'regularization_flag':True
    ,'regularization_coefficient':1e-5
    ,'class':aux.OMP_Model
    ,'colored':False
    ,'rain_dataset':False
    , 'get_extra_info_flag': False

}
parameters_config['denoising']['advanced'] = {
    'crop_size':(56,56)
    ,'train_set_settings':{
        'data':['CBSD432/','exploration_database_and_code/pristine_images/']  #TODO: Fix
        ,'read_from_hardisk':True
    }
    ,'batch_size':8
    ,'epochs':300
    ,'learning_rate':0.002
    ,'enable_scheduler':True
    , 'scheduler': {'step_size':20,'factor':0.5}
    ,'evaluation_cycle': (2,1)
    ,'model_parameters':{
            'initial_mode':'DCT'
            ,'early_stopping':False
            ,'learned_weighted_averaging':False
            ,'add_DC_atom':True
            ,'apply_attention_mode':True
    }
    ,'regularization_flag':True
    ,'regularization_coefficient':1e-5
    ,'class':aux.OMP_Model_Global_MultiScale
    ,'colored':False
    ,'rain_dataset':False
    , 'get_extra_info_flag': False
}

parameters_config['deraining'] = {
    'crop_size':(52,52)
    ,'train_set_settings':{
        'data':['RainTrainL/']  #TODO: Fix
        ,'read_from_hardisk':False
    }
    ,'batch_size':8
    ,'epochs':4000
    ,'learning_rate':0.002
    ,'enable_scheduler':True
    , 'scheduler': {'step_size':400,'factor':0.5}
    ,'evaluation_cycle': (200,20)
    ,'model_parameters':{
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
    ,'regularization_flag':True
    ,'regularization_coefficient':1e-2
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

    # sigma_noise = 15
    sigma_noise = 25
    # sigma_noise = 50
    print(sigma_noise)
    sigma_noise = float(sigma_noise)/255

    # choose config
    mode = 'basic'
    # mode = 'advanced'

    config = parameters_config[application][mode]

    train_composed = transforms.Compose(
        [aux.RandomCrop(config['crop_size']), aux.AddWhiteGaussianNoise(sigma_noise),
         aux.ToTensor(device, dtype), aux.Normlize()])

    test_composed = transforms.Compose([aux.AddWhiteGaussianNoise(sigma_noise),
                                        aux.ToTensor(device, dtype), aux.Normlize()])

    loss = aux.Log_Loss_2()
    print(mode)

if(application=='deraining'):
    testset_location = ['RainData/Rain100L/']
    testset2_location = ['RainData/test12/']


    sigma_noise = 4
    sigma_noise = float(sigma_noise) / 255

    config = parameters_config[application]

    train_composed = transforms.Compose(
        [aux.RandomCrop(config['crop_size']), aux.ToTensor(device, dtype),
         lambda sample: dict(list(sample.items()) + [('mean', 0)])])
    test_composed = transforms.Compose([aux.ToTensor(device, dtype),
         lambda sample: dict(list(sample.items()) + [('mean', 0)])])

    loss = aux.Tmp_Loss()




model_save_location =''






config['model_parameters']['sigma_noise']=sigma_noise
print(application)
print('parameters config:')
print(config)




train_set = aux.ImageDataSet(config['train_set_settings']['data'], transform=train_composed
                         ,read_immediatley_from_hardisk=config['train_set_settings']['read_from_hardisk']
                            ,colored=config['colored']
                            ,rain_dataset=config['rain_dataset']
                         )

print('train samples = ',len(train_set))
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
model.set_enable_cholesky_decomposition_flag(True)
model.setRegularization(config['regularization_flag'],config['regularization_coefficient'])
model.get_extra_info_flag(config['get_extra_info_flag'])

aux.train_model(train_set, test_set, model,'LGM',
            criterion=loss,batch_size=config['batch_size'],learning_rate=config['learning_rate']
            ,epochs=config['epochs']
            ,test_dataset2=test_set2
            # ,load_model=True
            ,enable_scheduler = config['enable_scheduler']
            ,scheduler_step = config['scheduler']['step_size'] if config['enable_scheduler'] else None
            ,scheduler_factor = config['scheduler']['factor'] if config['enable_scheduler'] else None
            ,eval_steps=config['evaluation_cycle']
            ,save_load_path = model_save_location
            )