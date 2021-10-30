FROM fzxu/nn

RUN mkdir actions-runner && cd actions-runner \
  && curl -o actions-runner-linux-x64-2.283.3.tar.gz -L https://github.com/actions/runner/releases/download/v2.283.3/actions-runner-linux-x64-2.283.3.tar.gz \
  && tar xzf ./actions-runner-linux-x64-2.283.3.tar.gz

ENV DEBIAN_FRONTEND=noninteractive

RUN cd /actions-runner && ./bin/installdependencies.sh

COPY scripts/config.sh /actions-runner/config.sh

COPY scripts/run.sh /actions-runner/run.sh

WORKDIR /actions-runner

ENTRYPOINT ./config.sh --url https://github.com/KevinXuxuxu/NN --token $TOKEN && ./run.sh
