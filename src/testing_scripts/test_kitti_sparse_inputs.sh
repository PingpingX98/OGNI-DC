GRU_iters=5
test_augment=0
optim_layer_input_clamp=100.0
depth_activation_format='linear'
depth_downsample_method='min'
pred_confidence_input=1


ckpt=/home/descfly/Projects/OGNI-DC/src/checkpoints/KITTI_generalization.pt

# for lidar_lines in 8 16 32 64
for lidar_lines in 64
do
  python main_time.py --dir_data /home/descfly/data/kitti_depth --data_name KITTIDC --split_json ../data_json/kitti_dc.json \
      --patch_height 352 --patch_width 1216 --lidar_lines $lidar_lines \
      --gpus 0 --max_depth 90.0  --test_crop \
      --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
      --depth_downsample_method $depth_downsample_method --pred_confidence_input $pred_confidence_input \
      --test_only --test_augment $test_augment --pretrain $ckpt \
      --log_dir /data/compare/metric/OGNI/experiments/ \
      --save "val_kitti_lines${lidar_lines}" \
      --save_result_only
  done