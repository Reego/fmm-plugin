cd plugin && /u/reego/blis-plugins-copy/share/blis/configure-plugin --build fmm_blis && make && cd ../

rm run_tests.x

make run_tests

./run_tests.x
