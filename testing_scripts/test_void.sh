GRU_iters=1
test_augment=0
optim_layer_input_clamp=1.0
depth_activation_format='exp'

ckpt=../checkpoints/NYU_best_performance.pt

for sample in 150 500 1500

do
  python main.py --dir_data ../datas/void/void_${sample} --data_name VOID \
    --gpus 0 --max_depth 10.0 \
    --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
    --test_only --test_augment $test_augment --pretrain $ckpt \
    --log_dir ../experiments \
    --save "test_void${sample}" \
    --save_result_only
done
