#!/bin/sh
python quantize.py trained/ai85-pebblesshells-qat8-q.pth.tar trained/ai85-pebblesshells-qat8-q.pth.tar --device MAX78000 -v "$@"
