FROM gitlab-registry.in2p3.fr/cta-lapp/cours/gray_scott_revolutions/grayscottbasecontainer/gray_scott_learning_platform_code_server:0.3.2@sha256:66d2a65f1415a07a7f9f600e4c73c947e7cd2b941c23fe2ac521a1f22c3eebdc

USER root

RUN mkdir -p /src \
 && chown -R $MAMBA_USER:$MAMBA_USER /src

USER $MAMBA_USER

RUN git clone --branch 5.1.1 https://github.com/kokkos/kokkos.git /src/kokkos
RUN git clone https://github.com/kokkos/kokkos-tools.git /src/kokkos-tools

RUN pixi global install --environment lecture binutils=2.45 cli11=2.6 cmake=3.31 cuda-nvcc=12.8 gcc=14.3 gxx=14.3 hdf5=1.14 make=4.4 && pixi clean cache --yes
