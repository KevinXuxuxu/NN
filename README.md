# NN

Deep learning framework written with Python/Numpy for learning purpose.

# Devops

This project has a docker image [fzxu/nn](https://hub.docker.com/repository/docker/fzxu/nn) built for developing environment. To start working on or testing the framework:

- (Prerequisite: docker should be installed) Run the `docker_run.sh` script in the root directory. Docker will pull the image from docker hub if it's your first time, which might take a while depending on your network.
    ```shell
    # docker run \
    #     -v `pwd`:/nn \
    #     -v `pwd`/bazel-cache:/root/.cache/bazel \
    #     -p 8888:8888 \
    #     --rm -it fzxu/nn /bin/bash
    ./docker_run.sh
    ```
    - This docker run command mount the repo dir and the bazel cache dir into the container, and also mapped 8888 port for jupyter notebook usesage.
- Once the container is started, you will be in `/nn` directory which is the mounting point of your repo dir. Now you can mess around with code, and use bazel:
    - To simply build the source code (which is usually not needed as there's no binary/executables in it)
    ```shell
    bazel build //src:nn_src
    
    # Output should look like:
    
    Extracting Bazel installation...
    Starting local Bazel server and connecting to it...
    INFO: Analyzed target //src:nn_src (6 packages loaded, 17 targets configured).
    INFO: Found 1 target...
    Target //src:nn_src up-to-date (nothing to build)
    INFO: Elapsed time: 13.164s, Critical Path: 0.11s
    INFO: 1 process: 1 internal.
    INFO: Build completed successfully, 1 total action
    ```
    - To run the tests (which will build the source code as dependency)
    ```shell
    bazel test //test:all
    
    # Output should look like:
    
    INFO: Analyzed 2 targets (19 packages loaded, 298 targets configured).
    INFO: Found 2 test targets...
    INFO: Elapsed time: 53.557s, Critical Path: 0.99s
    INFO: 10 processes: 6 internal, 4 processwrapper-sandbox.
    INFO: Build completed successfully, 10 total actions
    //test:layer_test                                                        PASSED in 0.8s
    //test:model_test                                                        PASSED in 0.8s

    Executed 2 out of 2 tests: 2 tests pass.
    There were tests whose specified size is too big. Use the --test_verbose_timeout_warnings command line option to see which
    ```
    - To test the framework by starting jupyter notebook
        - First build custom jupyter entry point with NN as dependency:
        ```shell
        bazel build //examples:jupyter

        # Output should look like:

        INFO: Analyzed target //examples:jupyter (1 packages loaded, 2 targets configured).
        INFO: Found 1 target...
        Target //examples:jupyter up-to-date:
          bazel-bin/examples/jupyter
        INFO: Elapsed time: 0.566s, Critical Path: 0.03s
        INFO: 4 processes: 4 internal.
        INFO: Build completed successfully, 4 total actions
        ```
        - Then run the built jupyter with some extra options:
        ```shell
        ./bazel-bin/examples/jupyter notebook --allow-root --ip 0.0.0.0

        # Output should look like:

        [I 11:45:45.372 NotebookApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
        [I 11:45:46.049 NotebookApp] Serving notebooks from local directory: /nn
        [I 11:45:46.049 NotebookApp] Jupyter Notebook 6.4.0 is running at:
        [I 11:45:46.049 NotebookApp] http://68abe5ca1832:8888/?token=a3fc01601b64f5242b32cb32e254c69456da60aa81ea9b90
        [I 11:45:46.049 NotebookApp]  or http://127.0.0.1:8888/?token=a3fc01601b64f5242b32cb32e254c69456da60aa81ea9b90
        [I 11:45:46.049 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
        [W 11:45:46.063 NotebookApp] No web browser found: could not locate runnable browser.
        [C 11:45:46.064 NotebookApp] 

            To access the notebook, open this file in a browser:
                file:///root/.local/share/jupyter/runtime/nbserver-933-open.html
            Or copy and paste one of these URLs:
                http://68abe5ca1832:8888/?token=a3fc01601b64f5242b32cb32e254c69456da60aa81ea9b90
             or http://127.0.0.1:8888/?token=a3fc01601b64f5242b32cb32e254c69456da60aa81ea9b90    # <--- only this link works
        ```
### TODO

- Transpose layer activation to be more intuitive
- Refactor to introduce operator concept (need design)
- Add more kinds of operators to implement/test more algorithms
    - Other activation functions
    - Other cost functions
    - Convolution layer
    - LSTM (?)
- Performance benchmark
- Better unit/integration testing (how?)
- ~~Introduce Bazel~~
    - ~~Build Bazel dependencies (e.g. python_rules) into docker image~~
- Investigate CI and devops
    - ~~Add docker support~~
    - explore CI options
- Remove torch/torchvision dependency from mnist example
