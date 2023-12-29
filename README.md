## Setup

Since github has the limit of 2GB of weights, I had to upload them to [Google Drive](https://drive.google.com/file/d/1-3Ll1_82AWtzPd2VuGlxPZZCCBlUXu4D/view?usp=sharing). Please, download the weights and put them to the `/weights` folder. 

## Work 

I began with the search of the models and the first one was [TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr). Despite the fact that it was good with English text, it has shown low performance with Chinese, so i decided to switch to the [Donut model](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Donut). 

In contrast to TrOCR, it was  better with Chinese symbols, so I decided to give it a try. But, since I had to upgrade the quality, I started looking for the datasets to teach my model on. Between those who required Chinese social networks accounts and those who had prehistoric firmats .mdb, I've finially managed to find [this one](https://github.com/GitYCC/traditional-chinese-text-recogn-dataset). Despite it being synthetic, it has shown good quality snd was available in an comfortable format, so I went eoth this one.

More details on the training phase could be found in the `/notebooks` section. 
