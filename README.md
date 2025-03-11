# cuFHE16384
cuFHE16384 is a modified version of [cuFHE](https://github.com/vernamlab/cuFHE) library that supports GBS for N = 16384. This library was made to compare performance of CPU and GPU with our FPGA-based accelerator with computational accuracies of 10 and 14 bits.  

# Installation
This library is tested on NVIDIA Tesla T4 and A100 GPUs.

## Build

1. Git clone
2. Create and change into build directory: `mkdir build && cd build`
3. Build by calling `cmake -DCMAKE_BUILD_TYPE=Release` \
    (Optional) Set cuda architecture to your GPU's compute capability by passing CMAKE_CUDA_ARCHITECTURES flag: `cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECUTRES=70`
4. Run generated Makefile by calling `make`

### Commands:
```bash
git clone https://github.com/eaglys-platform/cuFHE16384.git
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release # -DCMAKE_CUDA_ARCHITECTURES=70
make
```

## Note on Scheme 2
You can run Scheme 2 by compiling with `SCHEME=2`, although it is currently unstable and may result in errors.

## Running GBS Test
Execute the command below in the root directory of your cloned cufhe16384 repository.
```bash
./cufhe/bin/test_api_gpu
```

## Parameters used in cuFHE16384
<table border="1" style="border-collapse: collapse; width: 60%;">
  <thead>
    <tr>
      <th>Variable Name</th>
      <th>Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LARGE_N</td>
      <td>16384</td>
    </tr>
    <tr>
      <td>SMALL_N</td>
      <td>800</td>
    </tr>
    <tr>
      <td>SMALL_K</td>
      <td>1</td>
    </tr>
    <tr>
      <td>SMALL_L</td>
      <td>5</td>
    </tr>
    <tr>
      <td>B_G_BIT</td>
      <td>6</td>
    </tr>
    <tr>
      <td>SMALL_T</td>
      <td>7</td>
    </tr>
    <tr>
      <td>B_KS_BIT</td>
      <td>2</td>
    </tr>
    <tr>
      <td>LWE_NOISE</td>
      <td>pow(2.0, -30)</td>
    </tr>
  </tbody>
</table>

# Benchmark
### GBS execution time comparison 

<table border="1" style="border-collapse: collapse; width: 60%;">
  <!-- CPU-based Platform -->
  <tr>
    <th rowspan="2" style="text-align:left;">CPU-based Platform</th>
    <th>Ryzen 9</th>
    <th>Apple M1</th>
  </tr>
  <tr>
    <td>3.97s</td>
    <td>30.8s</td>
  </tr>
  
  <!-- GPU-based Platform -->
  <tr>
    <th rowspan="3" style="text-align:left;">GPU-based Platform</th>
    <th>Tesla T4</th>
    <th>A100</th>
  </tr>
  <tr>
    <td>617ms (Scheme 1)</td>
    <td>731ms (Scheme 1)</td>
  </tr>
  <tr>
    <td>754ms (Scheme 2)</td>
    <td></td>
  </tr>
  
  <!-- FPGA-based Platform -->
  <tr>
    <th style="text-align:left;">FPGA-based Platform</th>
    <td>250ms</td>
    <td></td>
  </tr>
</table>

# Citation

# References


[CGGI16]: Chillotti, I., Gama, N., Georgieva, M., & Izabachene, M. (2016, December). Faster fully homomorphic encryption: Bootstrapping in less than 0.1 seconds. In International Conference on the Theory and Application of Cryptology and Information Security (pp. 3-33). Springer, Berlin, Heidelberg.

[CGGI17]: Chillotti, I., Gama, N., Georgieva, M., & Izabachène, M. (2017, December). Faster Packed Homomorphic Operations and Efficient Circuit Bootstrapping for TFHE. In International Conference on the Theory and Application of Cryptology and Information Security (pp. 377-408). Springer, Cham.

[Dai15]: Dai, W., & Sunar, B. (2015, September). cuHE: A homomorphic encryption accelerator library. In International Conference on Cryptography and Information Security in the Balkans (pp. 169-186). Springer, Cham.
