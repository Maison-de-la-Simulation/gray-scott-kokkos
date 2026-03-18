#!/bin/bash
cd build
#time ./gray-scott.exe $* 1920 1080 1000 32
hyperfine --warmup 1  \
  -p 'rm -f /dev/shm/gray-scott.h5 && sync'  \
  "./gray-scott.exe $* 1920 1080 1000 32 && sync"
echo
