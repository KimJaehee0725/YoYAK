import argparse

# from infer import summarize_infer
from infer_models.load_kobigbird import mask_to_replace
from infer_models.load_koelectra import mask_overlap_concat, mask_overlap_average

def infer_chain(input_str, masking_model = "overlap_concat", masking_threshold = 0.2):

    if masking_model == "overlap_concat":
        from infer_models.load_koelectra import mask_overlap_concat
        mask_generate_infer = mask_overlap_concat

    elif masking_model == "overlap_average":
        from infer_models.load_koelectra import mask_overlap_average
        mask_generate_infer = mask_overlap_average

    elif masking_model == "non-overlap":
        from infer_models.load_koelectra import mask_per_510
        mask_generate_infer = mask_per_510

    mask_restore_infer = mask_to_replace

    # summarized_str = summarize_infer(input_str)
    masked_str = mask_generate_infer(input_str, masking_threshold)
    restored_str = mask_restore_infer(masked_str)
    return restored_str 


def main(args) :
    final_result = infer_chain(**args)
    print(final_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let's summarize")
    parser.add_argument("--masking_model", default = "overlap_concat", type = str, choices = ["overlap_concat", "overlap_average", "non-overlap"], help = "the model used for mask generation")
    parser.add_argument("--masking_threshold", default = 0.2, type = float, help = "the threshold for logits of masking model")
    args = parser.parse_args()
    main(args)