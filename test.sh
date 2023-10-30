CUDA_VISIBLE_DEVICES=1 \
python test.py \
--config config.yaml \
--testroot ./data \
--save_to test_results \
--resume ./checkpoint.pt