#!/usr/bin/env bash

mkdir -p weight_parts
rm -rf weight_parts/* && cp weight.tar weight_parts
cd weight_parts
split -b 90M -a 2 weight.tar
rm -rf weight.tar

