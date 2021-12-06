docker buildx build --platform linux/amd64 -t fzxu/nn-amd64 -f Dockerfile_amd64 .
docker buildx build --platform linux/arm64 -t fzxu/nn-arm64 -f Dockerfile_arm64 .
docker push fzxu/nn-amd64
docker push fzxu/nn-arm64
docker manifest create fzxu/nn fzxu/nn-amd64 fzxu/nn-arm64 --amend
docker manifest push fzxu/nn
