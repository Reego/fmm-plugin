# fmm-blis
Strassen and other FMM implementations using the BLIS Framework


To build fmm-blis:

1. Create the makefiles for the plugin based on your installation of BLIS. 
```
$ cd plugin
$ <path to installed blis>/share/blis/configure-plugin --build fmm_blis
$ make
```
2. To build the `fmm_blis` library, from the top level directory

```
$ make lib
```
3. To build the testsuite, from the top level directory

```
$ make test
```

---
---
# How to create your own plugin

Plugin--need a kernel (computational or packing) not supported by BLIS

Compile this for all the different architectures blis with right opt flags.

# To get existing examples:
`./configure-plugin <plugin-name>`

Gives a sample plugin that can be build into a library and use. 

Directories
- config  
    : every configuration our installed version of BLIS knows about. Each directory has those files from BLIS that define that configuration. What makefile flags to use. A place *.c we could register optimized kernels for that architecture. 
    A *.h file to Define things like default Mr and NR block size we want to use. 

- ref_kernels 
    : Reference kernels

- kernels: 


in Header, we have to have a kernel ID that the application uses to find the kernels that we have written. Prototypes of kernels.

Must register kernels before you can use the kernels. 


bli_plugin_register_fmm_blis
|_bli_plugin_init_fmm_blis_haswell_ref
