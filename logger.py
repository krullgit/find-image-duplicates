import logging.config
import os
import yaml

def setup_logger(output_file, logging_config):
    
    # logging_config must be a dictionary object specifying the configuration
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
        
    # load the logger config yml
    with open(logging_config, 'r') as fh:
        logging_config = yaml.safe_load(fh)
    
    if output_file is not None:
        logging_config['handlers']['file_handler']['filename'] = output_file
    
    logging.config.dictConfig(logging_config)
    
    
