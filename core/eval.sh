SR_PATH="/home/intern/ztw/Evaluation/LatentDehazing/NhHaze_selftest_399"
HR_PATH="/home/intern/ztw/Data/NhHaze_additional_test/gt"
IS_DIFF="false"

# PSNR
python eval_nr_indicator.py -m PSNR -r $HR_PATH -i $SR_PATH -is_diff $IS_DIFF

# SSIM
python eval_nr_indicator.py -m SSIM -r $HR_PATH -i $SR_PATH -is_diff $IS_DIFF

# ciede
python ciede.py -sr $SR_PATH -hr $HR_PATH -is_diff $IS_DIFF

# NIQE
python eval_nr_indicator.py -m NIQE -i $SR_PATH -is_diff $IS_DIFF

# MUSIQ
python eval_nr_indicator.py -m MUSIQ -i $SR_PATH -is_diff $IS_DIFF
