GRU_iters=5
test_augment=0
optim_layer_input_clamp=1.0
depth_activation_format='exp'
ckpt=../checkpoints/NYU_best_performance.pt
sample=500

for noise_level in 0.05 0.04 0.03 0.02 0.01

#for sample in 500
do
    python main.py --dir_data ../datas/nyudepthv2 --data_name NYU --split_json ../data_json/nyu.json \
        --gpus 0 --max_depth 10.0 --num_sample $sample \
        --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
        --test_only --test_augment $test_augment --pretrain $ckpt \
        --log_dir ../experiments/${sample}/ \
        --num_masks ${num_masks}
        --save "test_nyu" \
        --noise_level $noise_level \
#        --save_result_only --test_single
done

#python main.py --dir_data ../datas/nyudepthv2 --data_name NYU --split_json ../data_json/nyu.json \
#        --gpus 0 --max_depth 10.0 --num_sample 500 \
#        --GRU_iters 5 --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
#        --test_only --test_augment $test_augment --pretrain $ckpt \
#        --log_dir ../experiments/${sample}/ \
#        --save "test_nyu"