# ⚠️ EXPERIMENTAL: Transcribe audio to any language w/ 🤗 Transformers

[Whisper](https://openai.com/research/whisper), released by OpenAI in late 2022, till date has a near-SoTA performance across English & Multi-lingual benchmarks. 

The model was trained to do two key speech recognition tasks:
1. Transcribe a given audio in its base language. i.e. take the audio in language "X" and transcribe it.
2. Directly translate an audio to English. i.e. take audio in language "X" and transcribe into English.

As the world grows more and more connected, the need for high quality content is ever-so-increasing. One of the ways to make content more accessible (specially audio), is by transcribing it into different languages, thereby ensuring that the knowledge is spread. ⚡️

The typical workflow for transcribing from audio in language "X" to another language is as follows:
1. Translate and transcribe the audio in language "X" to English. (Base Whisper behaviour)
2. Translate the transcriptions from language "X" to another language. (Typically done with a LLM, you can use for example GPT-3.5/4)

This works great! However, as with any other process, the more steps you run, the higher the chances for error creep.

## Could we transcribe from language "X" to "Y" in one step?

*TL;DR* - Yes! Keep in mind that this is a hack, but it seems to work pretty well in our tests! These notes describe how to do it, but serious use of the technique would have to be validated much more throughly! This is because the model wasn't trained on the task we'll use it for, so results may not be as reliable.

Alright, let's get to it! Let's first try to transcribe an audio in english (`en`) language to german (`de`), italian (`it`), spanish (`es`), dutch (`nl`) and french (`fr`).

For a more interactive experience you can follow along with this colab! <a target="_blank" href="https://colab.research.google.com/github/Vaibhavs10/translate-with-whisper/blob/main/whisper_en_to_any_transcription.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Note: This tutorial assumes that you have run `huggingface-cli login` or used `notebook_login()` to authenticate with the hub. We only need this to access Common Voice, you can safely ignore this step if you run inference on your own audio files or public datasets.

```python
!pip -q install transformers datasets
```

Let's instantiate our speech recognition pipeline! For the purpose of this demo we'll use the Whisper-medium checkpoint, I'd recommend you use the [Whisper-large-v2](https://huggingface.co/openai/whisper-large-v2) checkpoint for more serious cases!

```python
from transformers import pipeline

whisper_asr = pipeline(
    "automatic-speech-recognition", model="openai/whisper-medium"
)
```

To keep things simple, we'll use the Common Voice dataset from the 🤗 Hub in `streaming` mode & resample the audio to 16KHz as expected by Whisper.

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

Time for the magic sauce! Here, we essentially force Whisper to decode in a specific language. Because Whisper was trained on 600K+ hours of data it is able to do so fairly well.

So the only change to make this happen would be to set the task as `transcribe` and change the target language.
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

Note: Some of these translations are a bit out of the line, however, we can fix these with the [Whisper-large-v2](https://huggingface.co/openai/whisper-large-v2) checkpoint and with some neat generation techniques like [constrastive search](https://huggingface.co/docs/transformers/generation_strategies#contrastive-search)!

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
Reading metadata...: 16354it [00:00, 100304.67it/s]
 Joe Keaton enthüftete Filme und Buster hatte Reservativen des Medien-Prozesses.
Reading metadata...: 16354it [00:00, 93428.54it/s]
 Joe Keaton disapprovato di film e Buster aveva riservazioni sui media.
Reading metadata...: 16354it [00:00, 56266.96it/s]
 Joe Keaton desapropó de filmes y Buster también tenía reservas sobre el medio.
Reading metadata...: 16354it [00:00, 90462.27it/s]
 Joe Keaton verliep de film en Buster had ook regeven over het media.
Reading metadata...: 16354it [00:00, 96229.24it/s]
 Joe Keaton s'enestime de la cinémathie et Buster a des réservations au sujet des médias.
```

Notice the subtle differences in the transcription, it still gets some of them wrong tho. For your actual use-case, I'd recommend tuning these parameters a bit or use one of the fine-tuned models on the hub.

Good luck! 🤝

## Bonus

This also means, that you can use the same transcriptions and get word & sentence level timestamps as well. 🔥
Check out this space here to know more!

## Next steps

1. Run a benchmark on [FLoRES](https://huggingface.co/datasets/facebook/flores) dataset.
2. Test the benchmark for [fine-tuned Whisper models](https://huggingface.co/models?other=whisper).

Help is more than welcome, IMO, open an issue/ PR and we can work together on this! 🤗
