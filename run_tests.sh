cd plugin && ~/blis-plugins-copy/share/blis/configure-plugin --build fmm_blis && make && cd ../

rm run_tests.x

make run_tests CXX=g++-13 CC=gcc-13

./run_tests.x