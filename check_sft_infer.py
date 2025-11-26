import os
import json
import soundfile as sf
import torchaudio
from tqdm import tqdm
from kimia_infer.api.kimia import KimiAudio

# --- 1. Load model ---
model_path = "/DATA/nfsshare/Adarsh/KIMI/output/finetuned_hf_for_inference"
tokenizer_path = "/DATA/nfsshare/Adarsh/KIMI/models/Tokenizers"
model = KimiAudio(model_path=model_path, tokenizer_path=tokenizer_path, load_detokenizer=True)

# --- 2. Sampling params ---
sampling_params = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

# --- 3. Paths ---
input_list_path = "/DATA/nfsshare/Adarsh/KIMI/datasets_earlier_used/data_with_semantic_codes_9l_test.txt"  # Your input file with .wav paths
output_audio_dir = "/DATA/nfsshare/Adarsh/KIMI/output/finetuned_hf_for_inference/audio_43k"
output_text_path = "/DATA/nfsshare/Adarsh/KIMI/output/finetuned_hf_for_inference/text_43k"
os.makedirs(output_audio_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_text_path), exist_ok=True)

print(f"📂 Output audio directory: {os.path.abspath(output_audio_dir)}")
print(f"📄 Output text path: {os.path.abspath(output_text_path)}")

# --- 4. Read list of files ---
with open(input_list_path, "r") as f:
    audio_paths = [line.strip() for line in f if line.strip().endswith(".wav")]#[:5]

# --- 5. Run inference ---
with open(output_text_path, "w") as txt_out:
    for idx, input_path in enumerate(audio_paths, 1):
        fname = os.path.basename(input_path)
        file_number = os.path.splitext(fname)[0]

        print(f"\n🎧 [{idx}/{len(audio_paths)}] Processing {fname}...")

        try:
            # Skip short/silent audio
            waveform, sr = torchaudio.load(input_path)
            duration_sec = waveform.shape[-1] / sr
            if duration_sec < 1.0:
                print(f"⚠️ Skipping {fname} (too short: {duration_sec:.2f}s)")
                continue

            # Prompt
            messages_conversation = [
                {"role": "user", "message_type": "text", "content": "Please listen to this Chinese audio and reply with its spoken English translation."},
                {"role": "user", "message_type": "audio", "content": input_path}
            ]

            # Generate both audio + text
            wav_output, text_output = model.generate(messages_conversation, **sampling_params, output_type="both")

            if not text_output or not text_output.strip():
                print(f"⚠️ No text generated for {fname}")
                continue

            # Save audio
            reply_filename = f"reply_audio_{file_number}.wav"
            reply_path = os.path.join(output_audio_dir, reply_filename)
            sf.write(reply_path, wav_output.detach().cpu().view(-1).numpy(), 24000)

            # Save text
            txt_out.write(json.dumps({
                "file": fname,
                "reply_audio": reply_filename,
                "reply_text": text_output.strip()
            }) + "\n")

            print(f"✅ Replied to {fname} → {reply_filename}")
            print(f"✅ Replied to {fname} → {text_output}")

        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")








