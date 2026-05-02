from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

class LocalTranslator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.translator = pipeline(
            "translation", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            src_lang="eng_Latn", # Default source
            device=device
        )

    def translate(self, text, target_lang_code="hin_Deva"):
        """Translates text locally using NLLB."""
        # Note: NLLB uses specific codes like 'hin_Deva' for Hindi
        result = self.translator(text, forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang_code])
        return result[0]['translation_text']