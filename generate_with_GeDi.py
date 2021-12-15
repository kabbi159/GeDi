
from transformers import GPT2Tokenizer
from modeling_gpt2 import GPT2LMHeadModel

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

output_dir = 'topic_small'
code_desired = "true"
code_undesired = "false"

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model.to('cuda')

gedi_model = GPT2LMHeadModel.from_pretrained(output_dir)
gedi_model.to('cuda')

# from train_GeDi import processors, output_modes

# features = torch.load('data/AG-news/cached_dev_gpt2-medium_192_sst-2')

# all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
# all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
# # all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
# all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
# dev_dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)

# dev_sampler = SequentialSampler(dev_dataset)
# dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=4)

#setting arguments for generation
#max generation length
gen_length = 200
#omega from paper, higher disc_weight means more aggressive topic steering
disc_weight = 30
#1 - rho from paper, should be between 0 and 1 higher filter_p means more aggressive topic steering
filter_p = 0.8
#tau from paper, preserves tokens that are classified as correct topic
target_p = 0.8
#hyperparameter that determines class prior, set to uniform by default
class_bias = 0

if gen_length>1024:
  length = 1024
else:
  length = gen_length

secondary_code = 'climate'

bpe_tokens = tokenizer.encode(secondary_code)

#Specify prompt below
prompt = "In a shocking finding"

start_len=0
text_ids = tokenizer.encode(prompt)
encoded_prompts=torch.LongTensor(text_ids).unsqueeze(0).to('cuda')

multi_code = tokenizer.encode(secondary_code)
attr_class = 1

generated_sequence = model.generate(input_ids=encoded_prompts,
                                         pad_lens=None,
                                          max_length=200,
                                          top_k=None,
                                          top_p=None,
                                          repetition_penalty= 1.2,
                                          rep_penalty_scale= 10,
                                          eos_token_ids = tokenizer.eos_token_id,
                                          pad_token_id = 0,
                                          do_sample= False,
                                          penalize_cond= True,
                                          gedi_model= gedi_model,
                                          tokenizer= tokenizer,
                                          disc_weight= disc_weight,
                                          filter_p = filter_p,
                                          target_p = target_p,
                                          class_bias = class_bias,
                                          attr_class = attr_class, # 1: (code_1 = pt_id, code_0 = nt_id)
                                          code_0 = code_undesired,
                                          code_1 = code_desired,
                                          multi_code=multi_code
                                          )

text = tokenizer.decode(generated_sequence.tolist()[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)
print('\n')
print(text)