'''
Author: Yorai Shaoul
Date: 2023-02-03

Example script for downloading using the TartanAir dataset toolbox.
'''

# General imports.
import sys

# Local imports.
sys.path.append('..')
import tartanair as ta

# Create a TartanAir object.
tartanair_data_root = '/ocean/projects/cis240055p/liuyuex/gemma/datasets/tartanair_auto'

ta.init(tartanair_data_root)

# Download data from following environments.
env = [ "Prison",
        "Ruins",
        "UrbanConstruction",
]

ta.download(env = env, 
              difficulty = ['easy'], 
              modality = ['image', 'depth'],  
              camera_name = ['lcam_front'], 
              unzip = True)

# Can also download via a yaml config file.
# ta.download(config = 'download_config.yaml')
