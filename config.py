import os

class Config():

    noise_size = 100
    batch_size = 256
    generator_feature=64
    discriminator_feature=64
    data_path = "path"

    path = os.path

    OUTPUT_DIR = 'output'
    FAKE_FACES_DIR = os.path.join(OUTPUT_DIR, 'fake_faces')

    _paths = [OUTPUT_DIR, FAKE_FACES_DIR]

    for p in _paths:
        if not os.path.exists(p):
            os.mkdir(p)

opt=Config()