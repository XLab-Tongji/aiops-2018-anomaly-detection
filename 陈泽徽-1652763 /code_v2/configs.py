# Model configuration for our Experiment

config_KPI = {}
config_KPI['ratio'] = 0.7
config_KPI['num_units'] = 256
config_KPI['dropout'] = 0.5
config_KPI['input_dim'] = 22
config_KPI['epoch_num'] = 15
config_KPI['batch_size'] = 128
config_KPI['augment_type'] = 'VAE'

config_KPI['optimizer'] = 'adam'
config_KPI['lr'] = 1e-3
config_KPI['dataset'] = 'KPI'
config_KPI['verbose'] = True
config_KPI['print_every'] = 500
config_KPI['adam_beta'] = 1e-3