import torch
import librosa
from functools import lru_cache
from typing import Dict, List
from kimia_infer.utils.data import KimiAContent
from kimia_infer.utils.special_tokens import instantiate_extra_tokens


class LazySupervisedDataset:
    """
    Dataset for speech-to-speech fine-tuning with audio-only supervision.
    Uses raw waveform directly (no Whisper feature extraction).
    """

    def __init__(self, raw_data_list, whisper_model, text_tokenizer, max_len: int, kimia_token_offset: int):
        super().__init__()
        self.raw_data = raw_data_list
        self.whisper_model = whisper_model   # kept for API compatibility (not used)
        self.text_tokenizer = text_tokenizer
        self.max_len = max_len
        self.kimia_token_offset = kimia_token_offset

        self.extra_tokens = instantiate_extra_tokens(self.text_tokenizer)
        self.pad_token = self.extra_tokens.pad

        print(f"Waveform-only dataset initialized with {len(raw_data_list)} samples.")

    def __len__(self):
        return len(self.raw_data)

    # -------------------------------------------------
    # 1️⃣ Waveform Extraction (your version exactly)
    # -------------------------------------------------
    def extract_whisper_feat(self, wav: str):
        wav = librosa.load(wav, sr=16000)[0]
        return wav

    # -------------------------------------------------
    # 2️⃣ Tokenize a Single Message (audio only)
    # -------------------------------------------------
    def tokenize_message(self, message: Dict, has_loss: bool):
        kimia_msg = KimiAContent()
        role = message["role"]

        # Start role tokens
        if role == "user":
            kimia_msg.audio_append(self.extra_tokens.kimia_user_msg_start)
        elif role == "assistant":
            kimia_msg.audio_append(self.extra_tokens.kimia_assistant_msg_start)

        if message.get("message_type", "audio") != "audio":
            raise ValueError("Speech-only dataset expects 'audio' message_type.")

        audio_path = message["content"]
        wav_feature = self.extract_whisper_feat(audio_path)
        kimia_msg.continuous_feature.append(wav_feature)

        kimia_msg.audio_append(self.extra_tokens.media_begin)

        # Use placeholder tokens
        speech_tokens = message.get("audio_tokens", [self.extra_tokens.kimia_text_blank])
        kimia_msg.audio_extend(speech_tokens, is_continuous=True, audio_token_loss_mask=has_loss)

        kimia_msg.audio_append(self.extra_tokens.media_end, audio_token_loss_mask=has_loss)
        return kimia_msg

    # -------------------------------------------------
    # 3️⃣ Tokenize Conversation
    # -------------------------------------------------
    def tokenize_conversation(self, conversation: List[Dict]) -> KimiAContent:
        msgs = []
        for msg in conversation:
            has_loss = msg["role"] == "assistant"
            msg_tok = self.tokenize_message(msg, has_loss)
            msgs.append(msg_tok)

        merged = msgs[0]
        for nxt in msgs[1:]:
            merged.merge(nxt)
        return merged

    # -------------------------------------------------
    # 4️⃣ __getitem__ for training
    # -------------------------------------------------
    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        item = self.raw_data[idx]
        conversation = item["conversation"]

        tokenized = self.tokenize_conversation(conversation)
        (
            audio_input_ids,
            _text_input_ids,  # unused
            is_continuous_mask,
            audio_token_loss_mask,
            _text_token_loss_mask,  # unused
        ) = tokenized.to_tensor()

        audio_features = tokenized.continuous_feature  # raw waveform

        # Next-token prediction labels (audio only)
        audio_labels = torch.cat(
            (audio_input_ids[:, 1:], audio_input_ids.new_full((1, 1), self.pad_token)), dim=1
        )
        audio_loss_mask = torch.cat(
            (audio_token_loss_mask[:, 1:], audio_token_loss_mask.new_full((1, 1), False)), dim=1
        )

        return dict(
            input_ids=audio_input_ids,
            whisper_input_feature=audio_features,  # actually waveform
            is_continuous_mask=is_continuous_mask,
            labels=(audio_labels, audio_loss_mask),
        )

    # -------------------------------------------------
    # 5️⃣ Collate Function
    # -------------------------------------------------
    @staticmethod
    def collate_fn(batch):
        assert len(batch) == 1, "Micro-batch size = 1 (adjust for larger batching if GPU allows)"
        return batch[0]

