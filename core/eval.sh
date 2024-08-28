HR_PATH="/home/intern/ztw/ztw/ztw/Data/Resized_OHaze"
SR_PATH="/home/intern/ztw/ztw/ztw/Evaluation/LatentDehazing/Finetune_ddim_OHaze_swin_transformer_resize_128_64_w_softmax_mask_71_3-v0_w1/samples"
IS_DIFF="false"



# psnr
python eval_nr_indicator.py -m PSNR -i $SR_PATH -r $HR_PATH -is_diff $IS_DIFF

# # ssim
python eval_nr_indicator.py -m SSIM -i $SR_PATH -r $HR_PATH -is_diff $IS_DIFF
python eval_nr_indicator.py -m CLIPIQA -i $SR_PATH -is_diff $IS_DIFF
python eval_nr_indicator.py -m NIMA -i $SR_PATH -is_diff $IS_DIFF
# # # NIQE
# # python eval_nr_indicator.py -m BRISQUE -i $SR_PATH -is_diff $IS_DIFF

# # # NIQE
# # python eval_nr_indicator.py -m MUSIQ -i $SR_PATH -is_diff $IS_DIFF

# # MUSIQ
python ciede.py -sr $SR_PATH -hr $HR_PATH -is_diff $IS_DIFF
# --config configs/LatentDehazingData/test_data.yaml --ckpt CKPT_PATH --outdir OUTDIR --skip_grid --ddpm_steps 200 --base_i 0 --seed 10000

