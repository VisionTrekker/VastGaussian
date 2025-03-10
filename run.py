import os
# Vast-GS使用--images指定图片路径有问题
# print('---------------------------------------------------------------------------------')
# cmd = f'python train_vast.py -s /data2/jtx/data/rubble \
#         --exp_name rubble_test \
#         --resolution 4 \
#         --eval --llffhold 83 \
#         --manhattan --platform "cc"\
#         --pos "25.607364654541 0.000000000000 -12.012700080872" \
#         --rot "0.923032462597 0.000000000000 0.384722054005 0.000000000000 1.000000000000 0.000000000000 -0.384722054005 0.000000000000 0.923032462597" \
#         --m_region 2 --n_region 2 \
#         --iterations 60_000'
# print(cmd)
# os.system(cmd)

# print('---------------------------------------------------------------------------------')
# cmd = f'python render.py -s /data2/jtx/data/rubble \
#         --exp_name rubble \
#         --resolution 4 \
#         --eval --llffhold 83 \
#         --manhattan \
#         --pos "25.607364654541 0.000000000000 -12.012700080872" \
#         --rot "0.923032462597 0.000000000000 0.384722054005 0.000000000000 1.000000000000 0.000000000000 -0.384722054005 0.000000000000 0.923032462597" \
#         --load_iteration 60_000 \
#         --sample_freq -1'
#print(cmd)
#os.system(cmd)

# print('---------------------------------------------------------------------------------')
# cmd = f'python metrics.py -m output/rubble'
# print(cmd)
# os.system(cmd)



# python train_vast.py -s /data2/jtx/data/rubble --exp_name rubble --manhattan --images images_2 -r 1 --pos 25.607364654541 0.000000000000 -12.012700080872 --rot 0.923032462597 0.000000000000 0.384722054005 0.000000000000 1.000000000000 0.000000000000 -0.384722054005 0.000000000000 0.923032462597 --m_region 3 --n_region 3 --num_gpus 4



# import os
# # Vast-GS使用--images指定图片路径有问题
# print('---------------------------------------------------------------------------------')
# cmd = f'python train_vast.py \
#         -s /data2/jtx/data/matrixcity \
#         --exp_name matrixcity \
#         -r 1 \
#         --eval --llffhold 83 \
#         --manhattan --platform "tj"\
#         --pos "0.000000000000 0.000000000000 0.000000000000" \
#         --rot "-90.000000000000 0.000000000000 0.000000000000" \
#         --m_region 2 --n_region 2 \
#         --iterations 60_000'
#print(cmd)
#os.system(cmd)


# linggongtang_2_1
# input_folder = "/data2/liuzhi/remote_data/dataset_reality/test/linggongtang_2_1"
# exp_name = "linggongtang_2_2_ex_0.2_vis_1.0"
# pos = '"-6.025578498840 0.000000000000 -8.141436576843"'
# rot = '"0.967983663082 0.000000000000 -0.251013278961 0.000000000000 1.000000000000 0.000000000000 0.251013278961 0.000000000000 0.967983663082"'
# m_region = 1
# n_region = 2
# iteration_1st = 30_000
# iteration_2nd = 7_000

# linggongtang_4_1
input_folder = "/data2/liuzhi/remote_data/dataset_reality/test/linggongtang_4_1"
exp_name = "linggongtang_4_4_ex_0.2_vis_1.0"
pos = '"9.106966018677 0.000000000000 -3.241944551468"'
rot = '"0.992140889168 0.000000000000 -0.125125899911 0.000000000000 1.000000000000 0.000000000000 0.125125899911 0.000000000000 0.992140889168"'
m_region = 2
n_region = 2
iteration_1st = 30_000
iteration_2nd = 10_000

# linggongtang_8_1
# input_folder = "/data2/liuzhi/remote_data/dataset_reality/test/linggongtang_8_1"
# exp_name = "linggongtang_8_8_ex_0.2_vis_1.0"
# pos = '"-29.747535705566 -1.310751080513 -10.745515823364"'
# rot = '"0.986465454102 0.002463806886 -0.163950830698 -0.003844148014 0.999959766865 -0.008102498017 0.163924276829 0.008623084985 0.9864352345477"'
# m_region = 4
# n_region = 2
# iteration_1st = 30_000
# iteration_2nd = 15_000

print('---------------------------------------------------------------------------------')
cmd = f'CUDA_VISIBLE_DEVICES=1,2,3 \
        python train_vast.py \
        -s {input_folder} \
        --exp_name {exp_name} \
        -r -1 \
        --iterations {iteration_1st} \
        --manhattan --platform cc \
        --pos {pos} \
        --rot {rot} \
        --m_region {m_region} --n_region {n_region} \
        --extend_rate 0.2 \
        --visible_rate 1.0'
# print(cmd)
# os.system(cmd)


# Finetune
print('---------------------------------------------------------------------------------')
cmd = f'CUDA_VISIBLE_DEVICES=3 \
        python tune.py \
        -s {input_folder} \
        --exp_name {exp_name} \
        -r -1 \
        --pretrained_ply output/{exp_name}/point_cloud/iteration_{iteration_1st}/point_cloud.ply \
        --iterations {iteration_2nd} \
        --test_iterations {iteration_2nd} \
        --save_iterations {iteration_2nd} \
        --manhattan --platform cc \
        --pos {pos} \
        --rot {rot}'
print(cmd)
os.system(cmd)


print('---------------------------------------------------------------------------------')
cmd = f'CUDA_VISIBLE_DEVICES=2 \
        python render.py -s {input_folder} \
        --exp_name {exp_name} \
        -r -1 \
        --load_iteration {iteration_1st + iteration_2nd} \
        --manhattan --platform cc \
        --pos {pos} \
        --rot {rot} \
        --sample_freq -1'
print(cmd)
os.system(cmd)


print('---------------------------------------------------------------------------------')
cmd = f'CUDA_VISIBLE_DEVICES=1 \
        python metrics.py \
        -m output/{exp_name}'
print(cmd)
os.system(cmd)