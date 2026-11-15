import time
import numpy as np
import resampy
import soundfile as sf
import requests
from io import BytesIO
from typing import Iterator

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

@register("tts", "gpt-sovits")
class SovitsTTS(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]): 
        text,textevent = msg
        tts_options = textevent.get('tts', {}) or {}
        ref_file = (tts_options.get('ref_audio_path') or tts_options.get('ref_file')
                    or self.opt.REF_FILE)
        ref_text = (tts_options.get('prompt_text') or tts_options.get('ref_text')
                    or self.opt.REF_TEXT)
        text_lang = tts_options.get('text_lang') or 'zh'
        prompt_lang = tts_options.get('prompt_lang') or text_lang
        self.stream_tts(
            self.gpt_sovits(
                text=text,
                reffile=ref_file,
                reftext=ref_text,
                language=text_lang,
                prompt_lang=prompt_lang,
                server_url=self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
                tts_options=tts_options,
            ),
            msg
        )

    def gpt_sovits(self, text, reffile, reftext, language, server_url, tts_options=None,
                   prompt_lang=None) -> Iterator[bytes]:
        start = time.perf_counter()
        tts_options = tts_options or {}
        req={
            'text':text,
            'text_lang':language,
            'ref_audio_path':reffile,
            'prompt_text':reftext,
            'prompt_lang':prompt_lang or language,
            'media_type':'ogg',
            'streaming_mode':True
        }
        # 软件音频：使用能力中心配置的 TTS 批次参数（zl-ai 会注入到 /human 的 tts 字段）
        try:
            if tts_options.get('batch_size') is not None:
                req['batch_size'] = max(1, min(20, int(tts_options['batch_size'])))
            if tts_options.get('fragment_interval') is not None:
                req['fragment_interval'] = max(0.0, min(0.3, float(tts_options['fragment_interval'])))
            if tts_options.get('text_split_method'):
                req['text_split_method'] = str(tts_options['text_split_method'])
            if tts_options.get('speed_factor') is not None:
                req['speed_factor'] = max(0.5, min(2.0, float(tts_options['speed_factor'])))
            if tts_options.get('seed') is not None:
                req['seed'] = int(tts_options['seed'])
            if tts_options.get('top_k') is not None:
                req['top_k'] = max(1, min(100, int(tts_options['top_k'])))
            if tts_options.get('top_p') is not None:
                req['top_p'] = max(0.01, min(1.0, float(tts_options['top_p'])))
            if tts_options.get('temperature') is not None:
                req['temperature'] = max(0.1, min(1.5, float(tts_options['temperature'])))
            if tts_options.get('repetition_penalty') is not None:
                req['repetition_penalty'] = max(1.0, min(2.5, float(tts_options['repetition_penalty'])))
        except (TypeError, ValueError):
            logger.warning('invalid tts options, fallback to defaults: %s', tts_options)
        # 默认去掉 SoVITS 的 0.3s 片段静音，避免长回复卡顿
        req.setdefault('fragment_interval', 0.05)
        logger.info(
            'gpt_sovits options: batch_size=%s, fragment_interval=%s, text_split_method=%s, speed_factor=%s, seed=%s, repetition_penalty=%s',
            req.get('batch_size'), req.get('fragment_interval'), req.get('text_split_method'),
            req.get('speed_factor'), req.get('seed'), req.get('repetition_penalty'),
        )
        # req["text"] = text
        # req["text_language"] = language
        # req["character"] = character
        # req["emotion"] = emotion
        # #req["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        # req["streaming_mode"] = True
        try:
            res = requests.post(
                f"{server_url}/tts",
                json=req,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"gpt_sovits Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=None): #12800 1280 32K*20ms*2
                logger.info('chunk len:%d',len(chunk))
                if first:
                    end = time.perf_counter()
                    logger.info(f"gpt_sovits Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
            #print("gpt_sovits response.elapsed:", res.elapsed)
        except Exception as e:
            logger.exception('sovits')

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def __create_bytes_stream32(self,byte_stream):
        stream, sample_rate = sf.read(byte_stream)
        stream = stream.astype(np.float32)
        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
        if sample_rate != 32000 and stream.shape[0] > 0:
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=32000)
        return stream

    def stream_tts(self,audio_stream,msg:tuple[str, dict]):
        text,textevent = msg
        first = True
        pending16 = None
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:
                # 全链路统一 16k：口型和播放共用同一份采样
                stream16 = self.__create_bytes_stream(BytesIO(chunk))
                if pending16 is not None and pending16.shape[0] > 0:
                    stream16 = np.concatenate((pending16, stream16))
                i16 = 0
                while stream16.shape[0] - i16 >= self.chunk:
                    eventpoint={}
                    if first:
                        eventpoint={'status':'start','text':text}
                        first = False
                    eventpoint.update(**textevent)
                    self.parent.put_audio_frame(stream16[i16:i16+self.chunk], eventpoint)
                    i16 += self.chunk
                # 分片末尾不足 20ms 的采样留到下一片再拼，避免逐片丢弃造成“吞字”
                pending16 = stream16[i16:] if stream16.shape[0] > i16 else None
        eventpoint={'status':'end','text':text}
        eventpoint.update(**textevent)
        rem16 = pending16 if pending16 is not None else np.zeros(0, dtype=np.float32)
        pad16 = np.zeros(max(0, self.chunk - rem16.shape[0]), dtype=np.float32)
        self.parent.put_audio_frame(np.concatenate((rem16, pad16)), eventpoint)
