hipcc -Wall -O1 -std=c++17 --offload-arch=gfx908  att.cpp -o att && ./att

