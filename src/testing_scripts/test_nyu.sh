GRU_iters=5
test_augment=0
optim_layer_input_clamp=1.0
depth_activation_format='exp'

ckpt=/home/descfly/Projects/OGNI-DC-main/src/checkpoints/NYU_best_performance.pt

# for sample in 5 50 100 200 300 400 500 
# for sample in 1 5 50 100 200 500 1000 5000 20000
# for sample in 1 5 50 100
# for sample in 300 400
for sample in 500

do
python main.py --dir_data /home/descfly/data/nyudepthv2 --data_name NYU --split_json ../data_json/nyu.json \
    --gpus 0 --max_depth 10.0 --num_sample $sample \
    --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
    --test_only --test_augment $test_augment --pretrain $ckpt \
    --log_dir /data/compare/metric/OGNI-DC/experiments/ \
    --save "test_nyu_sample${sample}_gaussian" --batch_size 1\
    --save_result_only
    # --save "test_nyu_8msk_sample${sample}" \
    # --save 'nyu_1.10' \
done

# do
# python main_time.py --dir_data /home/descfly/data/nyudepthv2 --data_name NYU --split_json ../data_json/nyu.json \
#     --gpus 0 --max_depth 10.0 --num_sample $sample --batch_size 1 \
#     --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp \
#     --depth_activation_format $depth_activation_format \
#     --test_only --test_augment $test_augment --pretrain $ckpt \
#     --log_dir /data/compare/metric/OGNI-DC/experiments/ \
#     --save "test_nyu_sample${sample}_inference" \
#     # --save_result_only
#     # --save 'nyu_1.10' \
# done
