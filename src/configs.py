import os

# Defaults reported in the final Medical Image Analysis paper.
PAPER_NUM_GENES = 200
PAPER_SCALE_DIMS = (1, 4, 8, 40, 100, 200)
PAPER_MAX_GENE_COUNT = 2000
PAPER_BATCH_SIZE = 64
PAPER_LEARNING_RATE = 1.0e-4
PAPER_SEED = 2021

# Named layouts used by the scale-sensitivity experiments.  ``paper`` is the
# six-scale configuration used for the main results.
SCALE_PRESETS = {
    'paper': PAPER_SCALE_DIMS,
    'k4': (1, 8, 40, 200),
    'k5': (1, 4, 20, 100, 200),
    'k7': (1, 2, 4, 8, 40, 100, 200),
    'single': (PAPER_NUM_GENES,),
    # Kept for command-line compatibility with the first public release.
    'flat': (1, 20, PAPER_NUM_GENES),
}

# Encoder feature dimensions per backbone
ENCODER_FEATURE_DIMS = {
    'uni': 1024,
    'conch': 512,
    'resnet18': 512,
}

DEFAULT_DATA_ROOT = os.environ.get('GENAR_DATA_ROOT', './data')

# The paper evaluates one held-out slide per dataset. It is not reused as a
# validation loader during training.
DATASETS = {
    'PRAD': {
        'dir_name': 'PRAD',
        'val_slides': '',
        'test_slides': 'MEND145',
        'recommended_encoder': 'uni'
    },
    'her2st': {
        'dir_name': 'her2st',
        'val_slides': '',
        'test_slides': 'SPA148',
        'recommended_encoder': 'uni'
    },
    'kidney': {
        'dir_name': 'kidney',
        'val_slides': '',
        'test_slides': 'NCBI697',
        'recommended_encoder': 'uni'
    },
    'mouse_brain': {
        'dir_name': 'mouse_brain',
        'val_slides': '',
        'test_slides': 'NCBI667',
        'recommended_encoder': 'uni'
    },
    'ccRCC': {
        'dir_name': 'ccRCC',
        'val_slides': '',
        'test_slides': 'INT2',
        'recommended_encoder': 'uni'
    }
}
