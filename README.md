# Face Detection With DeepStream For Jetson Orin Nano

## DeepStream installation
### Download DeepStream SDK for Jetson Orin Nano
- Check for DeepStream SDK's existence:
    ```bash
    deepstream-app --version
    ```

- Follow the instructions in [NVIDIA DeepStream SDK documentation](https://docs.nvidia.com/metropolis/deepstream/7.1/index.html) for installation, in this project I use DeepStream 7.1 to be compatible with Jetpack 6.1.

- Check again if the installation success:
    ```bash
    deepstream-app --version
    ```
 
### Test with the DeepStream sample
- List the DeepStream sample config files:
    ```bash
    ls -l /opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepsteam-app || true
    ```
- Here I choose an example which take a file as an input, make a copy of the sample config:
    ```bash
    cp /opt/nvidia deepstream/deepstream-7.1/samples/configs/deepstream-app/source2_1080p_dec_infer-resnet_demux_int8.txt ~/deepstream_text.txt
    ```
- Open the new file:
    ```bash
    nano ~/deepstream_text.txt
    ```
    // Modify it to be compatible with your Jetson, in this situation, I change the input video path and use on-screen display, software encoder cause Jetson Orin Nano does not have NVENC engine.

## Project installation
Overall, DeepStream requires 2 files: model-level config file and pipeline level config file. Beside, we need to create a custom parser for Ultraface which handles processing output coordinates to bounding boxes.

### Create a custom parser for Ultraface
- Check for `nvdsinfer_customparser` folder:
    ```bash
    ls /opt/nvidia/deepstream/deepstream-7.1/sources/libs/nvdsinfer_customparser
    cd /opt/nvidia/deepstream/deepstream-7.1/sources/libs/nvdsinfer_customparser
    ```
- Create new file: `nvdsinfer_custom_ultraface.cpp`.
- Copy the code that I provide to that file.
- Add your .cpp file into the SRCFILES list in
    `/opt/nvidia/deepstream/deepstream-7.1/sources/libs/nvdsinfer_customparser/Makefile`

    - Like this:

        `SRCFILES:= nvdsinfer_custombboxparser.cpp \
                    nvdsinfer_customclassifierparser.cpp \
                    nvdsinfer_customsegmentationparser.cpp \
                    nvdsinfer_custom_ultraface.cpp`

- Make:
    ```bash
    cd /opt/nvidia/deepstream/deepstream-7.1/sources/libs/nvdsinfer_customparser
    sudo make
    ```
    - If you get this issues, follow the instruction below:
        `ubuntu@tegra-ubuntu:/opt/nvidia/deepstream/deepstream-7.1/sources/libs/nvdsinfer_customparser$ sudo make
        g++ -o libnvds_infercustomparser.so nvdsinfer_custombboxparser.cpp nvdsinfer_customclassifierparser.cpp nvdsinfer_customsegmentationparser.cpp -Wall -std=c++11 -shared -fPIC -I../../includes -I /usr/local/cuda-/include -Wl,--start-group -lnvinfer -Wl,--end-group
        In file included from /usr/include/aarch64-linux-gnu/NvInferLegacyDims.h:22,
                        from /usr/include/aarch64-linux-gnu/NvInfer.h:21,
                        from ../../includes/nvdsinfer_custom_impl.h:117,
                        from nvdsinfer_custombboxparser.cpp:15:
        /usr/include/aarch64-linux-gnu/NvInferRuntimeBase.h:24:10: fatal error: cuda_runtime_api.h: No such file or directory
        24 | #include <cuda_runtime_api.h>
        |          ^~~~~~~~~~~~~~~~~~~~
        compilation terminated.
        In file included from /usr/include/aarch64-linux-gnu/NvInferLegacyDims.h:22,
                        from /usr/include/aarch64-linux-gnu/NvInfer.h:21,
                        from ../../includes/nvdsinfer_custom_impl.h:117,
                        from nvdsinfer_customclassifierparser.cpp:15:
        /usr/include/aarch64-linux-gnu/NvInferRuntimeBase.h:24:10: fatal error: cuda_runtime_api.h: No such file or directory
        24 | #include <cuda_runtime_api.h>
        |          ^~~~~~~~~~~~~~~~~~~~
        compilation terminated.
        In file included from /usr/include/aarch64-linux-gnu/NvInferLegacyDims.h:22,
                        from /usr/include/aarch64-linux-gnu/NvInfer.h:21,
                        from ../../includes/nvdsinfer_custom_impl.h:117,
                        from nvdsinfer_customsegmentationparser.cpp:16:
        /usr/include/aarch64-linux-gnu/NvInferRuntimeBase.h:24:10: fatal error: cuda_runtime_api.h: No such file or directory
        24 | #include <cuda_runtime_api.h>
        |          ^~~~~~~~~~~~~~~~~~~~
        compilation terminated.
        make: *** [Makefile:34: libnvds_infercustomparser.so] Error 1`

    - Check for CUDA (double check with jetson_release)
        ```bash
        ls /usr/local
        ```
        // should see like this: `cuda, cuda-12.6`

    - Check for cuda-runtime-api.h existance:
        ```bash
        ls /usr/local/cuda/include/cuda_runtime_api.h
        ```
        // should see like this:`/usr/local/cuda/include/cuda_runtime_api.h`

    - Fix the Makefile path, open the Makefile in that directory, find the line:

        `CFLAGS+= -I../../includes \
                -I /usr/local/cuda-$(CUDA_VER)/include`

        - Case A: if your folder is /usr/local/cuda, change to:

            `CFLAGS+= -I../../includes \
                    -I /usr/local/cuda/include`
        - Case B: if your folder is /usr/local/cuda-12.6, change to:

            `CUDA_VER?=12.6
            CFLAGS+= -I../../includes \
                    -I /usr/local/cuda/include` 

- Rebuild
    ```bash    
    sudo make clean # this will delete the old file libnvds_infercustomparser.so
    sudo make # should see like this: g++ -o libnvds_infercustomparser.so
    ```
- Verify
    ```bash
    ls -lh libnvds_infercustomparser.so 
    # should see like this: rwxr-xr-x 1 root root 222K Oct 23 16:13 libnvds_infercustomparser.so

    sudo make install 
    # Once this work, you can add your Ultraface parser
    ```
## Note
- Input URI must be started with: file://…
    - Ex: file:///home/ubuntu/projects/video1.mp4
- Consider to use the absolute path for “config_file"
