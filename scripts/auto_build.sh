
cd ./modeling/layers/deform_attn

if [ -d ./build ];then
   rm -rf ./build ./dist ./MultiScaleDeformableAttention.egg-info
fi
sh ./make.sh

cd ../diff_ras
if [ -d ./build ];then
   rm -rf ./build ./dist ./rasterizer.egg-info
fi
python setup.py build install
