docker run \
    -v `pwd`:/nn \
    -v `pwd`/bazel-cache:/root/.cache/bazel \
    -p 8888:8888 \
    --rm -it fzxu/nn /bin/bash
