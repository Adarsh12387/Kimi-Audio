from torch.utils.data import Dataset
from functools import lru_cache
import torch
from typing import Dict, List
import librosa

from kimia_infer.utils.special_tokens import instantiate_extra_tokens
from kimia_infer.utils.data import KimiAContent


class LazySupervisedDataset(Dataset):
    """
    Dataset for speech-to-speech training (with optional user text).
    - Loss is applied only to target (assistant) audio.
    - User text is used as conditioning only.
    - Source audio is used as conditioning only.
    """

    def __init__(self, raw_data_list: List[Dict], whisper_model, text_tokenizer, max_len: int, kimia_token_offset: int):
        super().__init__()
        self.raw_data = raw_data_list
        self.whisper_model = whisper_model
        self.text_tokenizer = text_tokenizer
        self.max_len = max_len
        self.kimia_token_offset = kimia_token_offset

        self.extra_tokens = instantiate_extra_tokens(self.text_tokenizer)
        self.pad_token = self.extra_tokens.pad

        print(f"Dataset loaded with {len(raw_data_list)} samples.")

    def __len__(self):
        return len(self.raw_data)

    def extract_whisper_feat(self, wav_path: str):
        """Extract continuous embeddings from Whisper model."""
        wav, sr = librosa.load(wav_path, sr=16000)
        wav_tensor = torch.tensor(wav).unsqueeze(0).to(torch.cuda.current_device())
        with torch.no_grad():
            features = self.whisper_model(wav_tensor)
        features = features.reshape(features.shape[0], features.shape[1] // 4, features.shape[2] * 4)
        return features.cpu()

    def _tokenize_text(self, text: str):
        return [] if text is None else self.text_tokenizer.encode(text, bos=False, eos=False)

    def tokenize_message(self, message: Dict, extract_whisper_feature: bool = False) -> KimiAContent:
        """Tokenize a single message to KimiAContent."""
        kimia_content_msg = KimiAContent()
        role = message.get("role")
        msg_type = message.get("message_type")
        content = message.get("content")
        is_target_audio = role == "assistant" and msg_type == "audio"

        # Text conditioning (user text only)
        if msg_type == "text" and role == "user":
            text_tokens = self._tokenize_text(content)
            kimia_content_msg.text_extend(text_tokens, has_loss=False)
            kimia_content_msg.audio_extend([self.extra_tokens.kimia_text_blank] * len(text_tokens))

        # Source or target audio
        elif msg_type == "audio" and content:
            audio_tokens = message.get("audio_tokens", [])
            kimia_content_msg.audio_append(self.extra_tokens.media_begin)
            kimia_content_msg.audio_extend(audio_tokens, is_continuous=True, audio_token_loss_mask=is_target_audio)
            kimia_content_msg.audio_append(self.extra_tokens.media_end, audio_token_loss_mask=is_target_audio)
            kimia_content_msg.text_extend([self.extra_tokens.kimia_text_blank] * (len(audio_tokens) + 2))

            if extract_whisper_feature:
                # Use precomputed feature if exists
                if "whisper_feat" in message:
                    whisper_feat = torch.tensor(message["whisper_feat"])
                else:
                    whisper_feat = self.extract_whisper_feat(content)
                kimia_content_msg.continuous_feature.append(whisper_feat)

        return kimia_content_msg

    def tokenize_conversation(self, messages: List[Dict]) -> KimiAContent:
        """Tokenize full conversation with source and target audio."""
        msgs = []
        for message in messages:
            msg_obj = self.tokenize_message(message, extract_whisper_feature=True)
            msgs.append(msg_obj)

        merged = msgs[0]
        for m in msgs[1:]:
            merged.merge(m)
        return merged

    @lru_cache(maxsize=None)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_entry = self.raw_data[idx]
        conv = data_entry["conversation"]
        tokenized_conv = self.tokenize_conversation(conv)

        audio_input_ids, text_input_ids, is_continuous_mask, audio_token_loss_mask, text_token_loss_mask = tokenized_conv.to_tensor()
        audio_features = tokenized_conv.continuous_feature

        # Only target audio contributes to loss
        audio_labels = audio_input_ids.masked_fill(~audio_token_loss_mask, self.pad_token)
        text_labels = text_input_ids  # user text used as conditioning only

        return {
            "input_ids": audio_input_ids,
            "text_input_ids": text_input_ids,
            "whisper_input_feature": audio_features,
            "is_continuous_mask": is_continuous_mask,
            "labels": (audio_labels, text_labels)
        }

    @staticmethod
    def collate_fn(batch):
        assert len(batch) == 1  # micro-batch size 1
        return batch[0]

