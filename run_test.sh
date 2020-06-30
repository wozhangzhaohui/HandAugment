#!/bin/bash
#source ~/zzh/env/bin/activate

dst_dir0=output/stage0
dst_dir1=output/stage1
result_path0=${dst_dir0}/result.txt
result_path1=${dst_dir1}/result.txt

# stage0
echo 'stage0 start'
python test_hands19task1.py \
  --output_result_path ${result_path0} \
  -v \
  --input_test_img_folder dataset/HANDS19_Challenge/Task1/test_images \
  --model_path weights/stage0.pth \
  --gpu_id 0 \
  --batch_size 256
cd ${dst_dir0}
zip -q result.txt.zip result.txt
cd -
echo 'stage0 done'

# hand region augment
echo 'hand region augmentation start'
# single thread
#python hand_region_augment.py \
#  --joint_list_path ${dst_dir0}/result.txt \
#  --dst_dir cache/hands19task1/test_images_augment \
#  -v
# multi thread
mkdir -p cache/part
split -l 5000 ${result_path0} cache/part/result.txt.part -d -a 2
for ((i = 0; i < 25; i += 1)); do
  index=$(echo $i | awk '{printf("%02d",$1)}')
  python hand_region_augment.py \
    -v \
    --dst_dir cache/hands19task1/test_images_augment \
    --joint_list_path cache/part/result.txt.part${index} &
done
wait
rm -rf cache/part
echo 'hand region augmentation done'

# stage1
echo 'stage1 start'
python test_hands19task1.py \
  --output_result_path ${result_path1} \
  -v \
  --input_test_img_folder cache/hands19task1/test_images_augment \
  --model_path weights/stage1.pth \
  --gpu_id 0 \
  --batch_size 256
cd ${dst_dir1}
zip -q result.txt.zip result.txt
echo 'stage1 done'
