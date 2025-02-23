autotrain llm --train \
    --project-name "llama3-autotrain" \
    --model "beomi/Llama-3-Open-Ko-8B" \
    --data-path "royboy0416/ko-alpaca" \
#    --data-path "/content/dataset" \
    --text-column "text" \
    --peft \
    --quantization "int4" \ # autotrain quantization 은 int4, int8 지원
    --lr 2e-4 \
    --batch-size 1 \
    --epochs 1 \
    --trainer sft \
    --model_max_length 256 \
    --save_total_limit 3 > train.out &