# ⚠️ EXPERIMENTAL: Transcribe audio to any language w/ 🤗 Transformers

[Whisper](https://openai.com/research/whisper), released by OpenAI in late 2022, till date has a near-SoTA performance across English & Multi-lingual benchmarks. 

The model was trained to do two key speech recognition tasks:
1. Transcribe a given audio in its base language. i.e. take the audio in language "X" and transcribe it.
2. Directly translate an audio to English. i.e. take audio in language "X" and transcribe in English.

As the world grows more and more connected, the need for high quality content is ever-so-increasing. One of the ways to make content, specially audio, more accessible is by transcribing it into different languages there by ensuring that the knowledge is still spread. ⚡️

The typical workflow for transcribing from audio in language "X" to another language is as follows:
1. Transcribe/ Translate the audio in language "X" to English. (Base Whisper behaviour)
2. Translate the transcriptions from language "X" to another language. (Typically done with a LLM, you can also use GPT-3.5/4)

This works great, however, as with any process, the more the number of steps, the more are chances for error creep.

## But, can we transcribe from language "X" to "Y" in one step?

*TL;DR* - Yes! Though, this is a hack, it does seems to work, however, needs to be validated much more throughly! That said, the model wasn't trained on this specific objective, so the results may not be as reliable.

Alright let's get to it! To demonstrate how this works, let's try to transcribe an audio in english (`en`) language to german (`de`), italian (`it`), spanish (`es`), dutch (`nl`) and french (`fr`).

Want a more interactive experience! Follow along with this colab! <a target="_blank" href="https://colab.research.google.com/github/Vaibhavs10/translate-with-whisper/blob/main/whisper_en_to_any_transcription.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Note: This tutorial assumes that you have run `huggingface-cli login` or using `notebook_login()` to authenticate with the hub, we only need it to access Common Voice. You can safely ignore it if you are running inference on different audio file/ dataset.

```python
!pip -q install transformers datasets huggingface_hub bitsandbytes accelerate
```

Let's instantiate our speech recognition pipeline. For the purpose of demo, we will use `bnb` and `accelerate` to load a Whisper-large-v2 checkpoint. If you have access to a larger GPU VRAM then remove the `model_kwargs` & `torch_dtype` 🤗 
```python
import torch
from transformers import pipeline

whisper_asr = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-large-v2",
    torch_dtype=torch.float16,
    device_map="auto", 
    model_kwargs=
     {
          "load_in_8bit": True
          }
    )
```

To keep things simple, we'll use the Common Voice dataset from the 🤗 Hub via `streaming` mode & resample the audio to 16KHz as expected by Whisper.
```python
from datasets import load_dataset
from datasets import Audio

common_voice_en = load_dataset("mozilla-foundation/common_voice_11_0", "en",
                               revision="streaming",
                               split="test",
                               streaming=True,
                               use_auth_token=True)

common_voice_en = common_voice_en.cast_column("audio",
                                              Audio(sampling_rate=16000))
```

Since we cannot render audio in markdown, let's take a look at the transcription.
```python
next(iter(common_voice_en))["sentence"]
```

output:
```
Reading metadata...: 16354it [00:00, 31433.60it/s]
'Joe Keaton disapproved of films, and Buster also had reservations about the medium.'
```

Let's create a wee list of languages to transcribe too.
```python
list_of_languages = ["de", "it", "es", "nl", "fr"]
```

## Magic sauce 🍝

We essentially force Whisper to decode in the specific language. Because Whisper was trained on 600K+ hours of data it is able to do so fairly well.

So the only change you'd need to make to make this happen would be to set the task as `transcribe` and change the target language.
```python
for lang in list_of_languages:
    whisper_asr.model.config.forced_decoder_ids = (
        whisper_asr.tokenizer.get_decoder_prompt_ids(
            language=lang,
            task="transcribe"
            )
        )
    print(whisper_asr(next(iter(common_voice_en))["audio"]["array"])["text"])
```

output:
```
Reading metadata...: 16354it [00:00, 33718.24it/s]
 Joe Keaton hat Filme verabschiedet und Buster hatte auch Reservations über die Medien.
Reading metadata...: 16354it [00:00, 27172.67it/s]
 Joe Keaton ha disapprovato i film e Buster ha anche delle riservazioni sui media.
Reading metadata...: 16354it [00:00, 41110.13it/s]
 Joe Keaton disaproveció de los filmes y Buster también tenía reservaciones sobre el medio.
Reading metadata...: 16354it [00:00, 39696.06it/s]
 Joe Keaton onverstaanbaar van de films en Buster had ook bewaarschuwingen over de media.
Reading metadata...: 16354it [00:00, 39813.59it/s]
 Joe Keaton a dénoncé les films et Buster avait des réservations sur le médium.
```

Voila! it works! We successfully transcribed an english audio to other languages.

Some of these translations are a bit out of the line, however, we can fix these with some neat generation techniques like [constrastive search](https://huggingface.co/docs/transformers/generation_strategies#contrastive-search)!

You can use contrastive search by providing `penalty_alpha` and `top_p` to the `generate_kwargs` in the pipeline. You can read more about it [here](https://huggingface.co/blog/introducing-csearch). 🤗 

```python
for lang in list_of_languages:
    whisper_asr.model.config.forced_decoder_ids = (
        whisper_asr.tokenizer.get_decoder_prompt_ids(
            language=lang,
            task="transcribe"
            )
        )
    print(whisper_asr(
        next(iter(common_voice_en))["audio"]["array"], 
        generate_kwargs = 
         {
              "penalty_alpha": 0.6, 
              "top_k": 5,
         }
        )["text"])
```

output:
```
Reading metadata...: 16354it [00:00, 39409.04it/s]
 Joe Keaton verabschiedete sich von Filmen und Buster hatte Regeltäusen über die Medien.
Reading metadata...: 16354it [00:00, 34203.76it/s]
 Joe Keaton disapprovò i film e Buster aveva anche riservazioni sui media.
Reading metadata...: 16354it [00:00, 24372.39it/s]
 Joe Keaton aprovechó de los filmes y Buster también tenía reservaciones sobre el medio.
Reading metadata...: 16354it [00:00, 41170.46it/s]
 Joe Keaton onvoldoende films en Buster had ook besluitingen over het medium.
Reading metadata...: 16354it [00:00, 23721.35it/s]
 Joe Keaton n'approuve pas les films et Buster avait également des préjugés sur le media.
```

Notice the subtle differences in the transcription, it still gets some things here and there wrong. For your actual use-case, I'd recommend tuning these parameters a bit or use one of the fine-tuned models on the hub.

Good luck! 🤝

## Bonus

This also means, that you can use the same transcriptions and get word & sentence level timestamps as well. 🔥
Check out this space here to know more!

## Next steps

1. Run a benchmark on [FLoRES](https://huggingface.co/datasets/facebook/flores) dataset.
2. Test the benchmark for [fine-tuned Whisper models](https://huggingface.co/models?other=whisper).

Help is more than welcome, IMO, open an issue/ PR and we can work together on this! 🤗