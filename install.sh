cd blis
./configure --prefix=/Users/rodrigobrandao/blis-plugins-copy auto && make -j && make install
cd ../plugin
~/blis-plugins-copy/share/blis/configure-plugin --build fmm_blis 
cd ../