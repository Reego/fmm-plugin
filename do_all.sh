cd plugin && ~/blis-plugins-copy/share/blis/configure-plugin --build fmm_blis && make && cd ../

rm test_strassen.x

make test CXX=g++-13 CC=gcc-13

cp ./test_strassen.x ./strassen.x

./test_strassen.x