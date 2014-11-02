#!/usr/bin/env python
# encoding: utf-8
"""
Simple wrapper for calling SuperFlux with the correct defaults values.

"""

from SuperFlux import parser, main

if __name__ == '__main__':
    # parse arguments
    args = parser(lgd=True, threshold=0.25)
    # and run the main SuperFlux program
    main(args)