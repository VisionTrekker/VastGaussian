import os

print('---------------------------------------------------------------------------------')
cmd = f'python train_vast.py -s /data2/jtx/data/rubble \
        --exp_name rubble \
        --manhattan \
        --images images_2 -r 1 \
        --pos "25.607364654541 0.000000000000 -12.012700080872" \
        --rot "0.923032462597 0.000000000000 0.384722054005 0.000000000000 1.000000000000 0.000000000000 -0.384722054005 0.000000000000 0.923032462597" \
        --m_region 3 --n_region 2 \
        --iterations 60_000'
print(cmd)
os.system(cmd)

print('---------------------------------------------------------------------------------')
cmd = f'python render.py -s /data2/jtx/data/rubble \
        --exp_name rubble \
        --manhattan \
        --images images_2 -r 1 \
        --pos "25.607364654541 0.000000000000 -12.012700080872" \
        --rot "0.923032462597 0.000000000000 0.384722054005 0.000000000000 1.000000000000 0.000000000000 -0.384722054005 0.000000000000 0.923032462597" \
        --load_iteration 60_000 \
        --sample_freq 8'
print(cmd)
os.system(cmd)

print('---------------------------------------------------------------------------------')
cmd = f'python metrics.py -m output/rubble'
print(cmd)
os.system(cmd)



# python train_vast.py -s /data2/jtx/data/rubble --exp_name rubble --manhattan --images images_2 -r 1 --pos 25.607364654541 0.000000000000 -12.012700080872 --rot 0.923032462597 0.000000000000 0.384722054005 0.000000000000 1.000000000000 0.000000000000 -0.384722054005 0.000000000000 0.923032462597 --m_region 3 --n_region 3 --num_gpus 4