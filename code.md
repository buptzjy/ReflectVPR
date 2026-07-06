cd /media/data/zhangjingyi/ReflectVPR




CUDA_VISIBLE_DEVICES=0,1 \
REFLECTVPR_FORCE_RESTART=0 \
REFLECTVPR_CLEAR_OUTPUT=0 \
REFLECTVPR_MAX_ROUNDS=1 \
REFLECTVPR_SAMPLE_NUM=200 \
REFLECTVPR_IMAGE_ROOT=/media/data1/chenshunpeng1/datasets/gsv_cities/Images \
REFLECTVPR_OUTPUT_ROOT=/media/data/zhangjingyi/ReflectVPR/output_0606_v5 \
REFLECTVPR_EXPERIENCE_JSON=/media/data/zhangjingyi/ReflectVPR/experience_bank_v0.json \
ICLIGHT_AUTO_START=0 \
python run.py --cities London,Phoenix,Osaka,PRS



REFLECTVPR_OUTPUT_ROOT=/media/data/zhangjingyi/ReflectVPR/output_0606_v5_parallel \
REFLECTVPR_PER_CITY_SAMPLE=50 \
REFLECTVPR_GPU_LIST=0,1,2,3 \
bash run_parallel_agents.sh