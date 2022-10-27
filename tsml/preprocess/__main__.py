import argparse
from datetime import datetime

from tsml.preprocess.bike_sharing.blocks import BikeSharingProcessBlock
from tsml.preprocess.walmart.blocks import WalmartProcessBlockZero, WalmartProcessBlockOne
from tsml.utils.config import Config

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c',
    '--client', 
    type=str, 
    required=True
)
parser.add_argument(
    '-b',
    '--block', 
    type=str, 
    required=False
)

if __name__ == '__main__':

    args = parser.parse_args()
    if args.client == 'walmart':
        config   = Config.from_json('config/walmart/preprocess.json')

        if args.block == 'zero':
            bs_block = WalmartProcessBlockZero(**config.block_zero)
            bs_block.process()

        elif args.block == 'one':
            for block in config.block_one.keys():
                bs_block = WalmartProcessBlockOne(**config.block_one[block])
                bs_block.process()
    
        else:
            bs_block = WalmartProcessBlockZero(**config.preprocess.block_zero)
            bs_block.process()

            for block in config.block_one.keys():
                bs_block = WalmartProcessBlockOne(**config.block_one[block])
                bs_block.process()
    
    
    elif args.client == 'bike-sharing':
        config   = Config.from_json('config/bike_sharing/preprocess.json')
        bs_block = BikeSharingProcessBlock(**config.block_zero)
        bs_block.process()