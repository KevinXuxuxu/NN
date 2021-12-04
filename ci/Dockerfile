FROM fzxu/nn

RUN adduser --disabled-password --gecos "" actions-runner

USER actions-runner

RUN cd /home/actions-runner \
  && curl -o actions-runner-linux-x64-2.284.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.284.0/actions-runner-linux-x64-2.284.0.tar.gz \
  && tar xzf ./actions-runner-linux-x64-2.284.0.tar.gz

USER root

ENV DEBIAN_FRONTEND=noninteractive

RUN cd /home/actions-runner && ./bin/installdependencies.sh

USER actions-runner

ENV DEBIAN_FRONTEND=noninteractive

COPY main.sh /home/actions-runner/main.sh

WORKDIR /home/actions-runner

ENTRYPOINT ./main.sh
