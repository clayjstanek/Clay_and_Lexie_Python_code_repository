# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:46:52 2026

@author: cstan
"""

"""
Objective

By the end of today, she should:  Understand variables and types, Use conditionals and loops, Write and call functions,
Manipulate lists and Feel like you can “make something
"""
import logging
import math
import os, sys
import time
from datetime import datetime, timezone
from pathlib import Path

def main():
    
    base_time = datetime.now(timezone.utc)
    start_time = base_time.strftime('_%Y-%m-%d_%H:%M:%S')
    start_time_log = base_time.strftime('_%Y-%m-%d')
    print(base_time)
    print(f'start_time: {start_time}')
    print(f'start_time_log: {start_time_log}')    
    
    p = Path(__file__).parent
    print(p)

    logs_path = p.joinpath('logs')
    if not logs_path.exists():
        Path.mkdir(logs_path, parents=True)

    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        filename=p.joinpath(logs_path, f'LexieLog_{start_time_log}.log'),
        format='{asctime} - {levelname} - {message}',
        datefmt='%Y-%m-%d %H:%M:%S',
        style='{',
        level=logging.INFO
    )
    logging.info('Logging Initialized')



if __name__ == "__main__":
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
    sys.exit(0)