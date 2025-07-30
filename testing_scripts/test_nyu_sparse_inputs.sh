GRU_iters=5
test_augment=0
optim_layer_input_clamp=1.0
depth_activation_format='exp'

ckpt=/home/descfly/Projects/OGNI-DC-main/src/checkpoints/NYU_generalization.pt

# for sample in 1 5 50 100 200 300 400 500 
for sample in 500
# for sample in 500 
do
  python main_memory.py --dir_data /home/descfly/data/nyudepthv2 --data_name NYU --split_json ../data_json/nyu.json \
      --gpus 0 --max_depth 10.0 --num_sample $sample \
      --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
      --test_only --test_augment $test_augment --pretrain $ckpt \
      --log_dir /data/result/OGNI-DC/experiments/  \
      --save 'nyu_1.02' \
      # --save "test_nyu_8msk_sample${sample}" \
      # --save_result_only
      # --save_full --save_pointcloud_visualization
done

# --dir_data /home/descfly/data/nyudepthv2 --data_name NYU --split_json ../data_json/nyu.json 
#       --gpus 0 --max_depth 10.0 --num_sample 500
#       --GRU_iters 5 --optim_layer_input_clamp 1.0 --depth_activation_format exp
#       --test_only --test_augment 0 --pretrain /home/descfly/Projects/OGNI-DC-main/src/checkpoints/NYU_generalization.pt
#       --log_dir /data/result/OGNI-DC/experiments/  
#       --save 'nyu_1.02' 

