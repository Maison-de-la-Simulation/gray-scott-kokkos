FROM gitlab-registry.in2p3.fr/cta-lapp/cours/gray_scott_revolutions/grayscottbasecontainer/gray_scott_interactive_jupyter:0.2.2@sha256:9e7379ab75441aef261c7b4ff50b8aea1d9be98e22697cdf8b9415bea64ef796

USER root

RUN mkdir -p /src \
 && chown -R $MAMBA_USER:$MAMBA_USER /src
RUN date -I > /version.txt

USER $MAMBA_USER

RUN git clone --branch 5.1.1 https://github.com/kokkos/kokkos.git /src/kokkos
RUN git clone https://github.com/kokkos/kokkos-tools.git /src/kokkos-tools && git -C /src/kokkos-tools checkout 3d33d0a783c5321d6981b3b4e43d3c157d2ca887
COPY exercises /home/$MAMBA_USER/Examples/exercises
COPY README.md /home/$MAMBA_USER/Examples/

RUN pixi global install --environment lecture binutils=2.45 cli11=2.6 cmake=3.31 gcc=14.3 gxx=14.3 hdf5=1.14 make=4.4 && pixi clean cache --yes
