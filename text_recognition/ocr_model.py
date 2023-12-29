import re
from pathlib import Path

import torch

from model.main_module import DonutModelPLModule
from PIL import Image

from transformers import VisionEncoderDecoderConfig
from transformers import DonutProcessor, VisionEncoderDecoderModel


class OCRModel:
    def __init__(self):
        image_size = [1280, 960]
        max_length = 60
        path_to_weights = '../weights/epoch=0-step=5400.ckpt'

        config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        config.encoder.image_size = image_size
        config.decoder.max_length = max_length

        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2", config=config)
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s_cord-v2>'])[0]
        self.module = DonutModelPLModule.load_from_checkpoint(path_to_weights,
                                                              config=config,
                                                              processor=processor,
                                                              model=model)
        self.max_length = 60

    def recognize_text(self, image: Path) -> str:
        """
        This method takes an image file as input and returns the recognized text from the image.

        :param image: The path to the image file.
        :return: The recognized text from the image.
        """
        img = Image.open(image).convert('RGB')

        pixel_values = self.module.processor(img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()

        decoder_input_ids = torch.full((1, 1), self.module.model.config.decoder_start_token_id)

        outputs = self.module.model.generate(pixel_values.unsqueeze(0),
                                             decoder_input_ids=decoder_input_ids,
                                             max_length=self.module.model.decoder.config.max_position_embeddings,
                                             early_stopping=True,
                                             pad_token_id=self.module.processor.tokenizer.pad_token_id,
                                             eos_token_id=self.module.processor.tokenizer.eos_token_id,
                                             use_cache=True,
                                             num_beams=1,
                                             bad_words_ids=[[self.module.processor.tokenizer.unk_token_id]],
                                             return_dict_in_generate=True,
                                             output_scores=True, )

        predictions = []
        for seq in self.module.processor.tokenizer.batch_decode(outputs.sequences):
            seq = (seq.replace(self.module.processor.tokenizer.eos_token, "")
                   .replace(self.module.processor.tokenizer.pad_token, ""))
            seq = re.sub(r"<.*?>", "", seq).strip()  # remove possible tokens
            predictions.append(seq)
        pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", predictions[0])
        return pred
