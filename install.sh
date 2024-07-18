cd blis
./configure --prefix=/u/reego/blis-plugins-copy auto && make -j && make install
cd ../plugin
/u/reego/blis-plugins-copy/share/blis/configure-plugin --build fmm_blis 
cd ../
