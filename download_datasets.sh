# Download CLEVR dataset
cd DATA_ROOT/CLEVR
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
unzip CLEVR_v1.0.zip


# Download ShapeStacks dataset
cd DATA_ROOT/ShapeStacks
wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-mjcf.tar.gz
wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-meta.tar.gz
wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-rgb.tar.gz
wget http://shapestacks-file.robots.ox.ac.uk/static/download/v1/shapestacks-iseg.tar.gz
tar xvzf shapestacks-meta.tar.gz
tar xvzf shapestacks-mjcf.tar.gz
tar xvzf shapestacks-rgb.tar.gz
tar xvzf shapestacks-iseg.tar.gz

# Download CLEVR-Tex dataset
cd DATA_ROOT/CLEVRTEX
wget https://thor.robots.ox.ac.uk/~vgg/DATA_ROOT/clevrtex/clevrtex_packaged/clevrtex_full_part1.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/DATA_ROOT/clevrtex/clevrtex_packaged/clevrtex_full_part2.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/DATA_ROOT/clevrtex/clevrtex_packaged/clevrtex_full_part3.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/DATA_ROOT/clevrtex/clevrtex_packaged/clevrtex_full_part4.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/DATA_ROOT/clevrtex/clevrtex_packaged/clevrtex_full_part5.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/DATA_ROOT/clevrtex/clevrtex_packaged/clevrtex_outd.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/DATA_ROOT/clevrtex/clevrtex_packaged/clevrtex_camo.tar.gz
tar -zxvf clevrtex_full_part1.tar.gz
tar -zxvf clevrtex_full_part2.tar.gz
tar -zxvf clevrtex_full_part3.tar.gz
tar -zxvf clevrtex_full_part4.tar.gz
tar -zxvf clevrtex_full_part5.tar.gz
tar -zxvf clevrtex_outd.tar.gz
tar -zxvf clevrtex_camo.tar.gz
# Download Flowers dataset