import os
import asyncio
import chainlit as cl
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

# ---------- Clients ----------
def aoai():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )

def transcribe_pcm16_16k(audio_bytes: bytes) -> str:
    """Blocking STT; safe to call in a worker thread."""
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    lang = os.getenv("SPEECH_RECOGNITION_LANGUAGE", "en-IN")
    if not key or not region:
        return ""

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_recognition_language = lang

    stream_format = speechsdk.audio.AudioStreamFormat(
        samples_per_second=16000, bits_per_sample=16, channels=1
    )
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format=stream_format)
    push_stream.write(audio_bytes)
    push_stream.close()

    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    try:
        result = recognizer.recognize_once_async().get()
    except Exception:
        return ""

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return (result.text or "").strip()
    return ""

# ---------- Chainlit ----------
@cl.on_chat_start
async def start():
    cl.user_session.set("audio_buf", [])
    cl.user_session.set("aoai", aoai())
    await cl.Message("🎤 Hold the mic (or press & hold **P**), speak, then release. I’ll transcribe and answer.").send()

@cl.on_audio_start
async def on_audio_start():
    # IMPORTANT: Accept the audio stream. Without this, the UI stays in "Connecting (P)"
    print("🔊 audio stream starting…")
    return True

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    # You should see these logs once the stream is accepted
    buf = cl.user_session.get("audio_buf") or []
    buf.append(chunk.data)
    cl.user_session.set("audio_buf", buf)
    print(f"chunk: {len(chunk.data)} bytes")

@cl.on_audio_end
async def on_audio_end():
    print("🛑 audio stream ended")
    buf = cl.user_session.get("audio_buf") or []
    cl.user_session.set("audio_buf", [])
    if not buf:
        await cl.Message("(No audio captured)").send()
        return

    pcm = b"".join(buf)

    # 1) Azure Speech → text
    transcript = await asyncio.to_thread(transcribe_pcm16_16k, pcm)
    if not transcript:
        await cl.Message("I couldn’t hear that clearly. Please try again.").send()
        return

    # 2) Azure OpenAI → reply
    client: AzureOpenAI = cl.user_session.get("aoai")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    resp = await asyncio.to_thread(
        client.chat.completions.create,
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": transcript},
        ],
    )
    answer = resp.choices[0].message.content

    # 3) Show both
    await cl.Message(content=f"**You said:** {transcript}\n\n{answer}").send()

@cl.on_message
async def on_text_message(msg: cl.Message):
    client: AzureOpenAI = cl.user_session.get("aoai")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    resp = await asyncio.to_thread(
        client.chat.completions.create,
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": msg.content},
        ],
    )
    await cl.Message(content=resp.choices[0].message.content).send()
