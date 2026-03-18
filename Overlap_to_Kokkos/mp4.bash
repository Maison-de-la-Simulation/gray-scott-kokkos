cd build
rm -f *.png
data-to-pics -i /dev/shm/gray-scott.h5 -o .
rm -f gray-scott.mp4
ffmpeg -framerate 60 -pattern_type glob -i "*.png" -c:v libx264 -pix_fmt yuv420p gray-scott.mp4
rm -f *.png
ls -l gray-scott.mp4

