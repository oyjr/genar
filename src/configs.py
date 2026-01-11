import os

# Encoder feature dimensions per backbone
ENCODER_FEATURE_DIMS = {
    'uni': 1024,
    'conch': 512,
    'resnet18': 512,
}

DEFAULT_DATA_ROOT = os.environ.get('GENAR_DATA_ROOT', './data')

# Dataset configuration including recommended validation/test slides
DATASETS = {
    'PRAD': {
        'dir_name': 'PRAD',
        'val_slides': 'MEND145',
        'test_slides': 'MEND145',
        'recommended_encoder': 'uni'
    },
    'her2st': {
        'dir_name': 'her2st',
        'val_slides': 'SPA148',
        'test_slides': 'SPA148',
        'recommended_encoder': 'uni'
    },
    'kidney': {
        'dir_name': 'kidney',
        'val_slides': 'NCBI697',
        'test_slides': 'NCBI697',
        'recommended_encoder': 'uni'
    },
    'mouse_brain': {
        'dir_name': 'mouse_brain',
        'val_slides': 'NCBI667',
        'test_slides': 'NCBI667',
        'recommended_encoder': 'uni'
    },
    'ccRCC': {
        'dir_name': 'ccRCC',
        'val_slides': 'INT2',
        'test_slides': 'INT2',
        'recommended_encoder': 'uni'
    }
}
