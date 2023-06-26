# ⚠️ EXPERIMENTAL: Transcribe audio to any language w/ 🤗 Transformers

[Whisper](https://openai.com/research/whisper), released by OpenAI in late 2022, till date has a near-SoTA performance across English & Multi-lingual benchmarks. 

The model was trained to do two key speech recognition tasks:
1. Transcribe a given audio in its base language. i.e. take the audio in language "X" and transcribe it.
2. Directly translate an audio to English. i.e. take audio in language "X" and transcribe in English.

As the world grows more and more connected, the need for high quality content is ever-so-increasing. One of the ways to make content, specially audio, more accessible is by transcribing it into different languages there by ensuring that the knowledge is still spread. ⚡️

The typical workflow for transcribing from audio in language "X" to another language right now is as follows:
1. Transcribe/ Translate the audio in language "X" to English. (Base Whisper behaviour)
2. Translate the transcriptions from language "X" to another language. (Typically done with a LLM, you can also use GPT-3.5/4)

This works great, however, as with any process, the more the number of steps, the more are chances for error creep.

## But, can we transcribe from language "X" to "Y" in one step?

*TL;DR* - Yes! the hack seems to work, however, needs to be validated much more throughly!

Heads-up, this is pretty much a hack, the model wasn't trained on this specific objective, so the results may not be as reliable.

Alright let's get to it. Let's first try to transcribe an audio in english (`en`) language to german (`de`), italian (`it`), spanish (`es`), dutch (`nl`) and french (`fr`).

You can also follow along with this linked Colab for a more interactive experience!

Let's set up our development environment! 

Note: This tutorial assumes that you have run `huggingface-cli login` or using `notebook_login()` to authenticate with the hub, we only need it to access Common Voice. You can safely ignore it if you are running inference on different audio file/ dataset.

```python
!pip -q install transformers datasets
```

Let's instantiate our 
```python
from transformers import pipeline

whisper_asr = pipeline(
    "automatic-speech-recognition", model="openai/whisper-medium"
)
```

To keep things simple, we'll use the Common Voice dataset from the 🤗 Hub via `streaming` mode & resample the audio to 16KHz as expected by Whisper.
```
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

Since we cannot render audio here, let's take a look at the transcription.
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

Time for the magic sauce, here, we essentially force Whisper to decode in the specific language. Because Whisper was trained on 600K+ hours of data it is able to do so fairly well.

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
Reading metadata...: 16354it [00:01, 14201.30it/s]
 Joe Keaton hat sich von Filmen entgegengegeben und Buster hatte auch Reservatoren für das Medien-Diagnostik.
Reading metadata...: 16354it [00:00, 42041.34it/s]
 Joe Keaton non si è approvato di film e Buster ha anche riservato il medio.
Reading metadata...: 16354it [00:00, 23966.34it/s]
 Joe Keaton no se acuerda de los filmes y Buster también tenía reservas sobre el medio.
Reading metadata...: 16354it [00:00, 46689.15it/s]
 Joe Keaton is uitgeproven van filmen en Buster had ook bepaalde bezoeken over het media.
Reading metadata...: 16354it [00:00, 36661.35it/s]
 Joe Keaton s'est dévoilé de la film et Buster avait aussi des réservations sur le milieu.
```

Voila! it works! We successfully transcribed an english audio to other languages.

## Next steps

1. Run a benchmark on [FLoRES](https://huggingface.co/datasets/facebook/flores) dataset.
2. Test the benchmark for [fine-tuned Whisper models](https://huggingface.co/models?other=whisper).

Help is more than welcome, IMO, open an issue/ PR and we can work together on this! 🤗