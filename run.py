import os
# Vast-GS使用--images指定图片路径有问题
print('---------------------------------------------------------------------------------')
cmd = f'python train_vast.py -s /data2/jtx/data/rubble \
        --exp_name rubble_test \
        --resolution 4 \
        --eval --llffhold 83 \
        --manhattan --platform "cc"\
        --pos "25.607364654541 0.000000000000 -12.012700080872" \
        --rot "0.923032462597 0.000000000000 0.384722054005 0.000000000000 1.000000000000 0.000000000000 -0.384722054005 0.000000000000 0.923032462597" \
        --m_region 2 --n_region 2 \
        --iterations 60_000'
print(cmd)
os.system(cmd)

print('---------------------------------------------------------------------------------')
cmd = f'python render.py -s /data2/jtx/data/rubble \
        --exp_name rubble \
        --resolution 4 \
        --eval --llffhold 83 \
        --manhattan \
        --pos "25.607364654541 0.000000000000 -12.012700080872" \
        --rot "0.923032462597 0.000000000000 0.384722054005 0.000000000000 1.000000000000 0.000000000000 -0.384722054005 0.000000000000 0.923032462597" \
        --load_iteration 60_000 \
        --sample_freq -1'
#print(cmd)
#os.system(cmd)

print('---------------------------------------------------------------------------------')
cmd = f'python metrics.py -m output/rubble'
# print(cmd)
# os.system(cmd)



# python train_vast.py -s /data2/jtx/data/rubble --exp_name rubble --manhattan --images images_2 -r 1 --pos 25.607364654541 0.000000000000 -12.012700080872 --rot 0.923032462597 0.000000000000 0.384722054005 0.000000000000 1.000000000000 0.000000000000 -0.384722054005 0.000000000000 0.923032462597 --m_region 3 --n_region 3 --num_gpus 4



import os
# Vast-GS使用--images指定图片路径有问题
print('---------------------------------------------------------------------------------')
cmd = f'python train_vast.py \
        -s /data2/jtx/data/matrixcity \
        --exp_name matrixcity \
        -r 1 \
        --eval --llffhold 83 \
        --manhattan --platform "tj"\
        --pos "0.000000000000 0.000000000000 0.000000000000" \
        --rot "-90.000000000000 0.000000000000 0.000000000000" \
        --m_region 2 --n_region 2 \
        --iterations 60_000'
#print(cmd)
#os.system(cmd)