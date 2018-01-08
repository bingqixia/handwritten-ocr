#!/bin/sh
eval "../preprocessing/extract $1"
eval "python predictor.py --bin best_model --img tmp/path_info_tmp.txt"
