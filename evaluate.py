import os
import json
import soundfile as sf
import torchaudio
from tqdm import tqdm
from kimia_infer.api.kimia import KimiAudio
import argparse
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import TreebankWordTokenizer

# Tokenizer
tokenizer_nltk = TreebankWordTokenizer()
smoothing_fn = SmoothingFunction().method1

# --------------------------------------------------------
# Extract dataset from JSONL
# --------------------------------------------------------
def extract_data(json_file):
    data = []
    with open(json_file, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    ref_audio = []
    ref_text = []

    for data_item in data:
        conversation = data_item.get("conversation", [])
        for turn in conversation:
            if turn.get("role") == "user" and turn.get("message_type") == "audio":
                ref_audio.append(turn["content"])
            if turn.get("role") == "assistant" and turn.get("message_type") == "text":
                ref_text.append(turn["content"])

    return ref_audio, ref_text


# --------------------------------------------------------
# Processing pipeline
# --------------------------------------------------------
def process(args):
    # Load KimiAudio model
    model = KimiAudio(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        load_detokenizer=True
    )

    # Extract dataset
    ref_audio_list, ref_text_list = extract_data(args.json_file)

    # Output directories
    output_audio_dir = os.path.join(args.output_dir, "audio_outputs")
    os.makedirs(output_audio_dir, exist_ok=True)

    predictions_jsonl_path = os.path.join(args.output_dir, "predictions.jsonl")
    predictions_tsv_path = os.path.join(args.output_dir, "predictions.tsv")
    bleu_json_path = os.path.join(args.output_dir, "final_bleu_score.json")

    # Initialize accumulators for BLEU
    all_hyp_tokens = []
    all_ref_tokens = []

    with open(predictions_jsonl_path, "w") as jsonl_out, \
         open(predictions_tsv_path, "w") as tsv_out:

        # Write TSV header
        tsv_out.write("file\treply_audio\treference_text\treply_text\tline_bleu\n")

        for idx, (input_path, reference_text) in enumerate(
            tqdm(list(zip(ref_audio_list, ref_text_list)),
                 desc="Processing dataset")
        ):

            fname = os.path.basename(input_path)
            file_number = os.path.splitext(fname)[0]

            try:
                # Load audio to check length
                waveform, sr = torchaudio.load(input_path)
                if waveform.shape[-1] / sr < 1.0:
                    print(f"Skipping short file: {fname}")
                    continue

                # Prompt for model inference
                messages = [
                    {"role": "user", "message_type": "text",
                     "content": "Please listen to this Chinese audio and reply with its spoken English translation."},
                    {"role": "user", "message_type": "audio", "content": input_path}
                ]

                # Model output (audio + text)
                wav_output, text_output = model.generate(
                    messages,
                    audio_temperature=0.8,
                    audio_top_k=10,
                    text_temperature=0.0,
                    text_top_k=5,
                    audio_repetition_penalty=1.0,
                    audio_repetition_window_size=64,
                    text_repetition_penalty=1.0,
                    text_repetition_window_size=16,
                    output_type="both"
                )

                if not text_output or not text_output.strip():
                    continue

                # Save generated audio
                reply_audio_filename = f"reply_{file_number}.wav"
                reply_audio_path = os.path.join(output_audio_dir, reply_audio_filename)
                sf.write(reply_audio_path, wav_output.detach().cpu().view(-1).numpy(), 24000)

                # Compute line-wise BLEU
                ref_tokens = tokenizer_nltk.tokenize(reference_text.lower())
                hyp_tokens = tokenizer_nltk.tokenize(text_output.lower())

                if ref_tokens and hyp_tokens:
                    line_bleu = corpus_bleu([[ref_tokens]], [hyp_tokens], smoothing_function=smoothing_fn)
                else:
                    line_bleu = 0.0

                # Accumulate for global BLEU
                all_ref_tokens.append([ref_tokens])
                all_hyp_tokens.append(hyp_tokens)

                # Write JSONL entry
                jsonl_out.write(json.dumps({
                    "file": fname,
                    "reply_audio": reply_audio_path,
                    "reference_text": reference_text,
                    "reply_text": text_output.strip(),
                    "line_bleu": line_bleu
                }) + "\n")

                # Write TSV entry
                tsv_out.write(
                    f"{fname}\t{reply_audio_path}\t{reference_text}\t{text_output.strip()}\t{line_bleu}\n"
                )

            except Exception as e:
                print(f"Error processing {fname}: {e}")

    # Compute final corpus BLEU
    if all_hyp_tokens:
        corpus_bleu_score = corpus_bleu(all_ref_tokens, all_hyp_tokens, smoothing_function=smoothing_fn)
    else:
        corpus_bleu_score = 0.0

    # Save corpus BLEU score
    with open(bleu_json_path, "w") as f:
        json.dump({"corpus_bleu": corpus_bleu_score}, f, indent=2)

    print(f"\nFinal Corpus BLEU: {corpus_bleu_score * 100:.2f}")
    print(f"Saved to: {bleu_json_path}")


# --------------------------------------------------------
# CLI entry
# --------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--json_file", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    process(args)

