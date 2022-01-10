<<<<<<< HEAD
import argparse

from infer import summarize_infer
from infer_models.load_kobigbird import mask_restore_infer_kobigbird, mask_restore_infer_kcbert
from infer_models.load_koelectra import mask_overlap_concat, mask_overlap_average

def infer_chain(masking_model, masking_threshold, restore_model):

    if masking_model == "overlap_concat":
        from infer_models.load_koelectra import mask_overlap_concat
        mask_generate_infer = mask_overlap_concat

    elif masking_model == "overlap_average":
        from infer_models.load_koelectra import mask_overlap_average
        mask_generate_infer = mask_overlap_average

    elif masking_model == "non-overlap":
        from infer_models.load_koelectra import mask_per_510
        mask_generate_infer = mask_per_510

    if restore_model == "kobigbird":
        mask_restore_infer = mask_restore_infer_kobigbird
        
    elif restore_model == "kcbert":
        mask_restore_infer = mask_restore_infer_kcbert


    input_str = input()
    summarized_str = summarize_infer(input_str)
    masked_str = mask_generate_infer(summarized_str, args.masking_threshold)
    restored_str = mask_restore_infer(masked_str)
    return(restored_str)


def main(args) :
    final_result = infer_chain(**args)
    print(final_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let's summarize")
    parser.add_argument("--masking_model", default = "overlap_concat", type = str, choices = ["overlap_concat", "overlap_average", "non-overlap"], help = "the model used for mask generation")
    parser.add_argument("--masking_threshold", default = 0.2, type = float, help = "the threshold for logits of masking model")
    parser.add_argument("--restore_model", default = "kobigbird", type = str, choices = ["kobigbird", "kcbert"], help = "the model used for restore masking")
    args = parser.parse_args()
    main(args)
=======
import argparse

from infer import summarize_infer
from infer_models.load_kobigbird import mask_restore_infer_kobigbird, mask_restore_infer_kcbert
from infer_models.load_koelectra import mask_overlap_concat, mask_overlap_average

def main(args):

    if args.masking_model == "overlap_concat":
        from infer_models.load_koelectra import mask_overlap_concat
        mask_generate_infer = mask_overlap_concat

    elif args.masking_model == "overlap_average":
        from infer_models.load_koelectra import mask_overlap_average
        mask_generate_infer = mask_overlap_average

    elif args.masking_model == "non-overlap":
        from infer_models.load_koelectra import mask_per_510
        mask_generate_infer = mask_per_510

    if args.restore_model == "kobigbird":
        mask_restore_infer = mask_restore_infer_kobigbird
        
    elif args.restore_model == "kcbert":
        mask_restore_infer = mask_restore_infer_kcbert


    input_str = input()
    summarized_str = summarize_infer(input_str)
    masked_str = mask_generate_infer(summarized_str, args.masking_threshold)
    restored_str = mask_restore_infer(masked_str)
    print(restored_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let's summarize")
    parser.add_argument("--masking_model", default = "overlap_concat", type = str, choices = ["overlap_concat", "overlap_average", "non-overlap"], help = "the model used for mask generation")
    parser.add_argument("--masking_threshold", default = 0.2, type = float, help = "the threshold for logits of masking model")
    parser.add_argument("--restore_model", default = "kobigbird", type = str, choices = ["kobigbird", "kcbert"], help = "the model used for restore masking")
    args = parser.parse_args()
    main(args)
>>>>>>> 42f08cdddbbcee4b841a73c81f8bac0d606d6639
