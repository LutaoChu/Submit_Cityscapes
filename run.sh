export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

yml=
cfg=configs/${yml}.yml
save_interval=2000
save_dir=saved_model/
echo $yml



mkdir -p ${save_dir}

# nohup python -u -m paddle.distributed.launch --log_dir $save_dir train.py \
# --config $cfg --use_vdl --save_dir $save_dir  \
# --save_interval  $save_interval \
# --num_workers 4 --do_eval \
# 2>&1 | tee  -a ${save_dir}/log \

# python  -m paddle.distributed.launch val.py --config $cfg \
# --model_path ${save_dir}/best_model/model.pdparams \
# --aug_eval \
# --flip_horizontal \
# --num_workers 4


root_dir=${save_dir}/testset_flip_predict_results     
img_dir=data/cityscapes/test.list                                                                                                                                                                                                                                                                                  
python -m paddle.distributed.launch predict.py --config $cfg \
--model_path ${save_dir}/best_model/model.pdparams \
--image_path ${img_dir} \
--save_dir ${root_dir} \
--aug_pred \
--flip_horizontal \
&& \
python convert_cityscapes_trainid2labelid.py --root_dir ${root_dir} \
&& \
cd ${root_dir} && zip -r convert_to_labelid.zip  convert_to_labelid/ && cd -

