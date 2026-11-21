import os
import queue
import threading
import time
import numpy as np
import soxr
import soundfile as sf
import requests
from io import BytesIO
from typing import Iterator

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

@register("tts", "gpt-sovits")
class SovitsTTS(BaseTTS):

    # ── 持久流式会话（TTS_SESSION_MODE=1 启用；默认走原有逐段请求）──────
    def render(self, quit_event):
        # 注意：BaseTTS.render 是"起线程后立刻返回"——avatar 的 render() 在它之后
        # 才会启动推理/渲染线程。这里必须保持同样的语义，否则会把整条视频管线卡住。
        if os.environ.get('TTS_SESSION_MODE', '0').strip() != '1':
            return super().render(quit_event)
        try:
            from .sovits_session import SovitsSession
        except Exception:
            logger.exception('[tts-session] import failed, fallback to legacy')
            return super().render(quit_event)
        threading.Thread(target=self._session_loop, args=(quit_event, SovitsSession),
                         name='tts-session', daemon=True).start()

    def _session_loop(self, quit_event, SovitsSession):
        logger.info('[tts-session] mode=enabled（一条连接连续出音频）')
        sess = None
        max_idle = float(os.environ.get('TTS_SESSION_MAX_IDLE', '120') or 120)
        try:
            while not quit_event.is_set():
                try:
                    msg = self.msgqueue.get(block=True, timeout=0.2)
                except queue.Empty:
                    # 会话常驻：TTS 暂时间隔（LLM 思考/换句）不关闭连接，
                    # 否则每停一次就要付一次首包延迟，接缝又回来了。
                    # 只在超过 TTS_SESSION_MAX_IDLE（默认 120s）无文本时回收。
                    if sess is not None and sess.idle_seconds() > max_idle:
                        sess.finish()
                        sess = None
                    continue
                text, textevent = msg
                if not str(text or '').strip():
                    continue
                tts_options = (textevent or {}).get('tts', {}) or {}
                if sess is None or not getattr(sess, '_alive', False):
                    sess = self._open_session(SovitsSession, tts_options)
                if sess is not None and not sess.send(text):
                    # 连接可能被服务端回收：重连后重发一次，再不行才退回旧路径
                    try:
                        sess.close()
                    except Exception:
                        pass
                    sess = self._open_session(SovitsSession, tts_options)
                    if sess is not None:
                        sess.send(text)
                if sess is None:
                    # 会话不可用：这一条退回旧的逐段请求，保证不断音
                    try:
                        self.txt_to_audio(msg)
                    except Exception:
                        logger.exception('[tts-session] legacy fallback failed')
        finally:
            if sess is not None:
                sess.finish()
            try:
                self.stop_tts()
            except Exception:
                pass
        logger.info('[tts-session] thread stop')

    def _open_session(self, cls, tts_options: dict):
        server = str(getattr(self.opt, 'TTS_SERVER', '') or '')
        if not server:
            return None
        url = server.replace('https://', 'wss://').replace('http://', 'ws://').rstrip('/') + '/tts/ws'
        ref_file = (tts_options.get('ref_audio_path') or tts_options.get('ref_file')
                    or self.opt.REF_FILE)
        ref_text = (tts_options.get('prompt_text') or tts_options.get('ref_text')
                    or self.opt.REF_TEXT)
        text_lang = tts_options.get('text_lang') or 'zh'
        query = {
            'text_lang': text_lang,
            'prompt_lang': tts_options.get('prompt_lang') or text_lang,
            'ref_audio_path': ref_file or '',
            'prompt_text': ref_text or '',
        }
        for k in ('batch_size', 'fragment_interval', 'text_split_method', 'speed_factor',
                  'seed', 'repetition_penalty'):
            if tts_options.get(k) is not None:
                query[k] = tts_options[k]
        session = cls(self, url, query)
        if not session.open():
            return None
        return session

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
            # 流式必须用 wav（首块是 WAV 头，后续是裸 PCM16），由本类自己解析。
            # 不能用 ogg：SoVITS 每个片段独立编码成一个 ogg，逐片解码会在分片边界
            # 产生编码器 priming/padding 的“咔哒/电流音”，实测每 100ms 一次。
            'media_type':'wav',
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
            # SoVITS 在 MACA 上偶发崩溃后由 run_sovits.sh 5s 内拉起；
            # 这里对“连不上”做短重试，避免一次重启就把当前这句 TTS 丢掉（现场表现为半句没声音）。
            res = None
            for attempt in range(3):
                try:
                    res = requests.post(
                        f"{server_url}/tts",
                        json=req,
                        stream=True,
                        timeout=(5, 120),
                    )
                    break
                except requests.exceptions.ConnectionError as ce:
                    if attempt >= 2:
                        raise
                    wait = 1.5 * (attempt + 1)
                    logger.warning(
                        "gpt_sovits connect failed (attempt %d/3), retry in %.1fs: %s",
                        attempt + 1, wait, ce,
                    )
                    time.sleep(wait)
            end = time.perf_counter()
            logger.info(f"gpt_sovits Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
                
            first = True
            # 流式 wav：首块 44 字节头 + 后续裸 PCM16。这里把头剥掉，只把 PCM 交给上层，
            # 采样率/声道数记在实例上供 stream_tts 使用。
            self._stream_rate = None
            self._stream_channels = 1
            header_buf = bytearray()
            pending_odd = b''
            for chunk in res.iter_content(chunk_size=None):
                if not chunk:
                    continue
                if first:
                    end = time.perf_counter()
                    logger.info(f"gpt_sovits Time to first chunk: {end-start}s")
                    first = False
                if self.state != State.RUNNING:
                    continue
                if self._stream_rate is None:
                    header_buf += chunk
                    if len(header_buf) < 44:
                        continue
                    if bytes(header_buf[:4]) != b'RIFF':
                        # 兼容旧版服务端（返回整段文件/ogg）：原样交给上层按文件解码
                        logger.warning('gpt_sovits stream is not RIFF wav, fallback to file decode')
                        data = bytes(header_buf)
                        header_buf.clear()
                        if data:
                            yield data
                        continue
                    rate = int.from_bytes(header_buf[24:28], 'little')
                    channels = int.from_bytes(header_buf[22:24], 'little') or 1
                    bits = int.from_bytes(header_buf[34:36], 'little')
                    self._stream_rate = rate
                    self._stream_channels = channels
                    if bits != 16:
                        logger.warning('gpt_sovits stream bits=%s, expected 16', bits)
                    logger.info('gpt_sovits stream wav: rate=%s channels=%s', rate, channels)
                    payload = bytes(header_buf[44:])
                    header_buf.clear()
                else:
                    payload = chunk
                # HTTP 分块可能把 16bit 样本切成半个，余下 1 字节留到下一块拼
                if pending_odd:
                    payload = pending_odd + payload
                    pending_odd = b''
                if len(payload) % 2:
                    pending_odd = payload[-1:]
                    payload = payload[:-1]
                if payload:
                    yield payload
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
            n_in = int(stream.shape[0])
            t0 = time.perf_counter()
            stream = soxr.resample(stream, sample_rate, self.sample_rate)
            cost = (time.perf_counter() - t0) * 1000.0
            logger.info('[conv] audio %d->%d samples=%d cost=%.1fms',
                        sample_rate, self.sample_rate, n_in, cost)

        return stream


    def stream_tts(self,audio_stream,msg:tuple[str, dict]):
        text,textevent = msg
        first = True
        pending16 = None
        pending_play = None
        rs16 = None
        rs16_rate = None
        rs_play = None
        rs_play_rate = None
        play_rate = int(getattr(self.parent, 'playback_rate', 48000) or 48000)
        play_chunk = int(getattr(self.parent, 'play_chunk', play_rate // (self.opt.fps*2)))
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:
                # 只解码一次：16k 给口型/ASR，原生/44.1k 给播放
                stream_rate = getattr(self, '_stream_rate', None)
                if stream_rate:
                    # 流式 wav：PCM16LE 裸数据，直接按样本解释（无编解码器边界损耗）
                    raw = np.frombuffer(chunk, dtype='<i2').astype(np.float32) / 32768.0
                    channels = getattr(self, '_stream_channels', 1) or 1
                    if channels > 1:
                        raw = raw.reshape(-1, channels)[:, 0]
                    sr = stream_rate
                else:
                    raw, sr = sf.read(BytesIO(chunk))
                    raw = raw.astype(np.float32)
                    if raw.ndim > 1:
                        raw = raw[:, 0]

                if sr == self.sample_rate:
                    stream16 = raw
                else:
                    if rs16 is None or rs16_rate != sr:
                        rs16 = soxr.ResampleStream(sr, self.sample_rate, 1, dtype='float32')
                        rs16_rate = sr
                    stream16 = rs16.resample_chunk(raw, last=False)

                if sr == play_rate:
                    stream_play = raw
                else:
                    if rs_play is None or rs_play_rate != sr:
                        rs_play = soxr.ResampleStream(sr, play_rate, 1, dtype='float32')
                        rs_play_rate = sr
                    stream_play = rs_play.resample_chunk(raw, last=False)

                if pending16 is not None and pending16.shape[0] > 0:
                    stream16 = np.concatenate((pending16, stream16))
                if pending_play is not None and pending_play.shape[0] > 0:
                    stream_play = np.concatenate((pending_play, stream_play))
                i16 = 0
                iplay = 0
                while (stream16.shape[0] - i16 >= self.chunk
                       and stream_play.shape[0] - iplay >= play_chunk):
                    eventpoint={}
                    if first:
                        eventpoint={'status':'start','text':text}
                        first = False
                    eventpoint.update(**textevent)
                    self.parent.put_audio_frame(
                        stream16[i16:i16+self.chunk], eventpoint,
                        data_play=stream_play[iplay:iplay+play_chunk])
                    i16 += self.chunk
                    iplay += play_chunk
                # 分片末尾不足 20ms 的采样留到下一片再拼，避免逐片丢弃造成“吞字”
                pending16 = stream16[i16:] if stream16.shape[0] > i16 else None
                pending_play = stream_play[iplay:] if stream_play.shape[0] > iplay else None
        # 句子结束：把流式重采样器的尾巴吐出来。soxr 是流式的，不 flush 会扣住最后
        # 几十毫秒（32k→48k 实测约 56ms），听感就是每句末尾被削掉一截。
        try:
            if rs16 is not None:
                tail = rs16.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
                if getattr(tail, 'size', 0):
                    pending16 = tail if pending16 is None else np.concatenate((pending16, tail))
            if rs_play is not None:
                tail = rs_play.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
                if getattr(tail, 'size', 0):
                    pending_play = tail if pending_play is None else np.concatenate((pending_play, tail))
        except Exception:
            logger.debug('resampler flush failed', exc_info=True)

        eventpoint={'status':'end','text':text}
        eventpoint.update(**textevent)
        rem16 = pending16 if pending16 is not None else np.zeros(0, dtype=np.float32)
        remp = pending_play if pending_play is not None else np.zeros(0, dtype=np.float32)
        # flush 后可能多出整帧：先把成对完整的 20ms 帧发完，再发补齐的最后一帧
        i16 = ipl = 0
        while (rem16.shape[0] - i16 >= self.chunk and remp.shape[0] - ipl >= play_chunk):
            self.parent.put_audio_frame(rem16[i16:i16+self.chunk], {},
                                        data_play=remp[ipl:ipl+play_chunk])
            i16 += self.chunk
            ipl += play_chunk
        rem16 = rem16[i16:] if rem16.shape[0] > i16 else np.zeros(0, dtype=np.float32)
        remp = remp[ipl:] if remp.shape[0] > ipl else np.zeros(0, dtype=np.float32)
        pad16 = np.zeros(max(0, self.chunk - rem16.shape[0]), dtype=np.float32)
        padp = np.zeros(max(0, play_chunk - remp.shape[0]), dtype=np.float32)
        self.parent.put_audio_frame(np.concatenate((rem16, pad16)), eventpoint,
                                    data_play=np.concatenate((remp, padp)))
