cd plugin && /Users/rodrigobrandao/blis-plugins-copy/share/blis/configure-plugin --build fmm_blis && make && cd ../

rm driver.x

make driver CXX=g++-13 CC=gcc-13