from infer import summarize_infer

from infer_models.load_kobigbird import mask_restore_infer

input_str = input()
summarized_str = summarize_infer(input_str)
masked_str = mask_generate_infer(summarize_str)
restored_str = mask_restore_infer(masked_str)
