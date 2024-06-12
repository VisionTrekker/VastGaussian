# train building
#python train_vast.py -s datasets/Mill_19/building --exp_name building --manhattan --pos -62.527942657471 0.000000000000 -15.786898612976 --rot 0.932374119759 0.000000000000 0.361494839191 0.000000000000 1.000000000000 0.000000000000 -0.361494839191 0.000000000000 0.932374119759

# train rubble
command="python train_multigpu.py -s ../datasets/Mill19/rubble --exp_name rubble --manhattan --pos 25.607364654541 0.000000000000 -12.012700080872 --rot 0.923032462597 0.000000000000 0.384722054005 0.000000000000 1.000000000000 0.000000000000 -0.384722054005 0.000000000000 0.923032462597 --num_gpus 2"
nohup $command > log-6.12 2>&1 & echo $! > run.pid