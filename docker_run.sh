docker run \
	-v $(pwd):/nn \
	-v $(pwd)/bazel-cache:/root/.cache/bazel \
	-w /nn \
	-p 8888:8888 \
	--rm -it fzxu/nn /bin/bash
