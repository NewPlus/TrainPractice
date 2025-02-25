from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "NewPlus/llama3-autotrain-ko"

# 토크나이저 불러오기 및 pad token 설정 (없을 경우)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()

# Prompt 메시지
messages = [
    {"role": "user", "content": "안녕하세요. 넌 누구야??"}
]

# apply_chat_template 호출
input_ids = tokenizer.apply_chat_template(
    conversation=messages, 
    tokenize=True, 
    add_generation_prompt=True, 
    return_tensors='pt',
    return_attention_mask=True  # 옵션은 있지만 실제로 딕셔너리로 반환되지 않는 경우가 있음
)

# input_ids는 이제 텐서이므로, attention_mask를 직접 생성합니다.
attention_mask = (input_ids != tokenizer.pad_token_id).long()

# 모델 생성
output_ids = model.generate(
    input_ids.to('cuda'),
    attention_mask=attention_mask.to('cuda')
)

# 생성된 응답 디코딩 (프롬프트 길이 이후 토큰만 사용)
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
print(response)
