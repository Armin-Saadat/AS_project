def get_config():
    """Get the hyperparameter configuration."""
    config = {}

    config['mode'] = "ssl"
    config['use_wandb'] = True
    config['use_cuda'] = True
    config['log_dir'] = "/AS_clean/AS_thesis/logs"
    config['model_load_dir'] = None  # required for test-only mode

    # Hyperparameters for dataset. 
    config['view'] = 'all'  # all/plax/psax
    config['flip_rate'] = 0.3
    config['label_scheme_name'] = 'all'
    # must be compatible with number of unique values in label scheme
    # will be automatic in future update
    config['num_classes'] = 4

    # Hyperparameters for bicuspid valve branch
    # config['bicuspid_weight'] = 1.0 # default is 1.0

    # Hyperparameters for Contrastive Learning
    config['cotrastive_method'] = 'CE'  # 'CE'/'SupCon'/'SimCLR'/Linear'
    config['feature_dim'] = 128
    config['temp'] = 0.1

    # Hyperparameters for models.
    config['model'] = "r2plus1d_18"  # r2plus1d_18/x3d/resnet50/slowfast/tvn/ava_r2plus1d_18
    config['pretrained'] = False
    config['restore'] = True
    config['loss_type'] = 'cross_entropy'  # cross_entropy/evidential/laplace_cdf/SupCon/SimCLR
    config['use_ava'] = False

    # Hyperparameters for training.
    config['batch_size'] = 6
    config['num_epochs'] = 40
    config['lr'] = 1e-4
    config['sampler'] = 'AS'  # imbalanced sampling based on AS/bicuspid/random

    return config
