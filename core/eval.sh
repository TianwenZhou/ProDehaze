SR_PATH="/home/intern/ztw/ztw/ztw/Evaluation/LatentDehazing/NhHazetest_full_epoch71"
HR_PATH="/home/intern/ztw/ztw/ztw/Data/Resized_NhHaze"
IS_DIFF="false"

# ciede
python ciede.py -sr $SR_PATH -hr $HR_PATH -is_diff $IS_DIFF

# psnr
python eval_nr_indicator.py -m PSNR -i $SR_PATH -r $HR_PATH -is_diff $IS_DIFF

# ssim
python eval_nr_indicator.py -m SSIM -i $SR_PATH -r $HR_PATH -is_diff $IS_DIFF

# NIQE
python eval_nr_indicator.py -m NIQE -i $SR_PATH -is_diff $IS_DIFF

# MUSIQ
python eval_nr_indicator.py -m MUSIQ -i $SR_PATH -is_diff $IS_DIFF
