# ⚠️ EXPERIMENTAL: Translate audio to any language w/ 🤗 Transformers

Whisper, released by OpenAI in 2022, till date has a near-SoTA performance across English & Multilingual benchmarks. 

Whisper was trained to do two key speech recognition tasks:
1. Transcribe a given audio in its base language. i.e. take the audio in language "X" and transcribe it.
2. Directly translate an audio to English. i.e. take audio in language "X" and transcribe in English.

As the world grows more and more connected, the need for high quality content is ever-so-increasing. One of the ways to make content, specially audio, more accessible is by transcribing it to different languages there by ensuring that the knowledge is still spread.

The typical workflow for transcribing from audio in language "X" to another language right now is as follows:
1. Transcribe/ Translate the audio in language "X" to English. (Base Whisper behaviour)
2. Translate the transcriptions from language "X" to another language. (Typically done with a LLM, you can also use GPT-3.5/4)

This works great, however, as with any process that exists, the more the number of steps needed to reach the end-result, the more the number of chances for error to creep.

## But, can we transcribe from language "X" to "Y" in one step?

TL;DR - Yes! the hack seems to work, however, needs to be validated much more throughly!

Heads-up, this is pretty much a hack, the model wasn't trained on this objective, so the results may not be as reliable.

Alright let's get to it.


## Next steps

1. Run a benchmark on FLoRES.
2. Test the benchmark for fine-tuned Whisper models.
