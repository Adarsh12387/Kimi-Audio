import torch
import librosa
from functools import lru_cache
from typing import Dict, List
from kimia_infer.utils.data import KimiAContent
from kimia_infer.utils.special_tokens import instantiate_extra_tokens
import numpy as np
import torchaudio

class LazySupervisedDataset:
    """
    Dataset for speech-to-speech fine-tuning with text-dependent supervision.
    Compatible with the KimiAContent class.
    """

    def __init__(self, raw_data_list, whisper_model, text_tokenizer, max_len: int, kimia_token_offset: int):
        super().__init__()
        self.raw_data = raw_data_list
        self.whisper_model = whisper_model
        self.text_tokenizer = text_tokenizer
        self.max_len = max_len
        self.kimia_token_offset = kimia_token_offset

        self.extra_tokens = instantiate_extra_tokens(self.text_tokenizer)
        self.pad_token = self.extra_tokens.pad

        print(f"Speech-to-Speech dataset initialized with {len(raw_data_list)} samples.")

    def __len__(self):
        return len(self.raw_data)

    # -----------------------------
    # 1️⃣ Feature Extraction
    # -----------------------------
    '''def extract_whisper_feat(self, wav):
        if isinstance(wav, str):
            wav, sr = librosa.load(wav, sr=16000)
        elif isinstance(wav, np.ndarray):
            sr = 16000
        elif isinstance(wav, torch.Tensor):
            wav = wav.squeeze().cpu().numpy()
            sr = 16000
        else:
            raise ValueError(f"Unsupported wav type: {type(wav)}")

        wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80
        )
        mel = mel_transform(wav_tensor)
        log_mel = torch.log10(mel + 1e-6)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

        log_mel = log_mel.to(torch.cuda.current_device())

        # 🔥 Fix the dimension order
        #log_mel = log_mel.transpose(1, 2)  # now [1, seq_len, 80]

        with torch.no_grad():
            print("Input to Whisper:", log_mel.shape)
            continuous_feature = self.whisper_model(log_mel)

        continuous_feature = continuous_feature.reshape(
            continuous_feature.shape[0],
            continuous_feature.shape[1] // 4,
            continuous_feature.shape[2] * 4,
        )

        return continuous_feature.squeeze(0).cpu()'''
        
    def extract_whisper_feat(self, wav: str):
        wav = librosa.load(wav, sr=16000)[0]
        # if isinstance(wav, str):
        #     wav = librosa.load(wav, sr=16000)[0]

        #     wav_tensor = torch.tensor(wav).unsqueeze(0)[:, :]
        # elif isinstance(wav, torch.Tensor):
        #     wav_tensor = wav
        # else:
        #     raise ValueError(f"Invalid wav type: {type(wav)}")

        # wav_tensor = wav_tensor.to(torch.cuda.current_device())
        # continous_feature = self.whisper_model(wav_tensor)
        # continous_feature = continous_feature.reshape(
        #     continous_feature.shape[0],
        #     int(continous_feature.shape[1] // 4),
        #     continous_feature.shape[2] * 4,
        # )
        #print(wav.shape)
        return wav
        
    # -----------------------------
    # 2️⃣ Text Tokenization
    # -----------------------------
    def _tokenize_text(self, text: str):
        if text is None:
            return []
        return self.text_tokenizer.encode(text, bos=False, eos=False)

    # -----------------------------
    # 3️⃣ Tokenize a Single Message
    # -----------------------------
    def tokenize_message(self, message: Dict, has_loss: bool):
        kimia_msg = KimiAContent()
        role = message["role"]

        # Handle message types properly
        msg_type = message.get("message_type", "audio")

        # Start role tokens
        if role == "user":
            kimia_msg.audio_append(self.extra_tokens.kimia_user_msg_start)
            kimia_msg.text_append(self.extra_tokens.kimia_text_blank)
        elif role == "assistant":
            kimia_msg.audio_append(self.extra_tokens.kimia_assistant_msg_start)
            kimia_msg.text_append(self.extra_tokens.kimia_text_blank)

        # Handle text messages (skip audio processing)
        if msg_type == "text":
            text_content = message.get("content", "")
            if text_content:
                text_tokens = self._tokenize_text(text_content)
                kimia_msg.text_extend(text_tokens, text_token_loss_mask=has_loss)
                kimia_msg.audio_extend([self.extra_tokens.kimia_text_blank] * len(text_tokens))
            kimia_msg.text_append(self.extra_tokens.kimia_text_eos, text_token_loss_mask=has_loss)
            kimia_msg.audio_append(self.extra_tokens.kimia_text_blank)
            return kimia_msg

        # Handle audio messages
        elif msg_type == "audio":
            audio_path = message["content"]
            whisper_feature = self.extract_whisper_feat(audio_path)
            kimia_msg.continuous_feature.append(whisper_feature)

            kimia_msg.audio_append(self.extra_tokens.media_begin)
            kimia_msg.text_append(self.extra_tokens.kimia_text_blank)

            speech_tokens = message.get("audio_tokens", [self.extra_tokens.kimia_text_blank])
            kimia_msg.audio_extend(speech_tokens, is_continuous=True, audio_token_loss_mask=has_loss)
            kimia_msg.text_extend([self.extra_tokens.kimia_text_blank] * len(speech_tokens))

            kimia_msg.audio_append(self.extra_tokens.media_end, audio_token_loss_mask=has_loss)
            kimia_msg.text_append(self.extra_tokens.kimia_text_blank)

            transcript = message.get("transcript", None)
            if transcript:
                text_tokens = self._tokenize_text(transcript)
                kimia_msg.text_extend(text_tokens, text_token_loss_mask=has_loss)
                kimia_msg.audio_extend([self.extra_tokens.kimia_text_blank] * len(text_tokens))
                kimia_msg.text_append(self.extra_tokens.kimia_text_eos, text_token_loss_mask=has_loss)
                kimia_msg.audio_append(self.extra_tokens.kimia_text_blank)
            return kimia_msg

        else:
            raise ValueError(f"Unsupported message_type: {msg_type}")

    # -----------------------------
    # 4️⃣ Tokenize Conversation
    # -----------------------------
    def tokenize_conversation(self, conversation: List[Dict]) -> KimiAContent:
        msgs = []
        for i, msg in enumerate(conversation):
            has_loss = msg["role"] == "assistant"
            msg_tok = self.tokenize_message(msg, has_loss)
            msgs.append(msg_tok)

        merged = msgs[0]
        for nxt in msgs[1:]:
            merged.merge(nxt)
        return merged

    # -----------------------------
    # 5️⃣ __getitem__ for training
    # -----------------------------
    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        item = self.raw_data[idx]
        conversation = item["conversation"]

        tokenized = self.tokenize_conversation(conversation)
        (
            audio_input_ids,
            text_input_ids,
            is_continuous_mask,
            audio_token_loss_mask,
            text_token_loss_mask,
        ) = tokenized.to_tensor()

        audio_features = tokenized.continuous_feature

        # Create shifted labels (next-token prediction)
        audio_labels = torch.cat((audio_input_ids[:, 1:], audio_input_ids.new_full((1, 1), self.pad_token)), dim=1)
        text_labels = torch.cat((text_input_ids[:, 1:], text_input_ids.new_full((1, 1), self.pad_token)), dim=1)
        audio_loss_mask = torch.cat((audio_token_loss_mask[:, 1:], audio_token_loss_mask.new_full((1, 1), False)), dim=1)
        text_loss_mask = torch.cat((text_token_loss_mask[:, 1:], text_token_loss_mask.new_full((1, 1), False)), dim=1)

        return dict(
            input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=audio_features,
            is_continuous_mask=is_continuous_mask,
            labels=(audio_labels, text_labels, audio_loss_mask, text_loss_mask),
        )

    # -----------------------------
    # 6️⃣ Collate Function
    # -----------------------------
    @staticmethod
    def collate_fn(batch):
        assert len(batch) == 4, "Micro-batch size = 1 (adjust for larger batching)"
        return batch[0]

