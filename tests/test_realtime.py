import queue
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.realtime import LatestQueue, expired
from src.stt.whisper_backend import WhisperBackend
from src.audio.vad import VADSegmenter
from src.state import PipelineState, PipelineStatus
from src.tts.piper_backend import TTSWorker


class RealtimeTests(unittest.TestCase):
    def test_balanced_profile_rejects_old_speech_before_synthesis(self):
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put({"text_de": "Hallo", "segment_start": 10.0})
        state = PipelineState(status=PipelineStatus.RUNNING)
        engine = Mock()
        worker = TTSWorker(incoming, outgoing, engine, state)
        def get_item(timeout):
            worker.stop()
            return {"text_de": "Hallo", "segment_start": 10.0}
        incoming.get = get_item
        with patch("time.monotonic", return_value=17.1):
            worker.run()
        engine.synthesize.assert_not_called()
        self.assertTrue(outgoing.empty())

    def test_tts_latency_includes_segmentation_and_queue_wait(self):
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put({"text_de": "Hallo", "segment_start": 10.0,
                      "segment_duration": 3.0, "stt_duration": 1.0,
                      "translation_duration": .5})
        state = PipelineState(status=PipelineStatus.RUNNING)
        engine = Mock()
        worker = TTSWorker(incoming, outgoing, engine, state, max_segment_age=12)
        def synthesize(text):
            worker.stop()
            return b"wav", .5
        engine.synthesize.side_effect = synthesize
        with patch("time.monotonic", return_value=18.0):
            worker.run()
        latency = outgoing.get_nowait()["latency"]
        self.assertEqual(latency.estimated_user_delay, 8.0)
        self.assertEqual(latency.total_pipeline, 2.0)

    def test_tts_drops_result_that_became_stale_during_synthesis(self):
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put({"text_de": "Hallo", "segment_start": 10.0})
        state = PipelineState(status=PipelineStatus.RUNNING)
        engine = Mock()
        worker = TTSWorker(incoming, outgoing, engine, state, max_segment_age=12)
        with patch("time.monotonic", return_value=20.0) as clock:
            def synthesize(text):
                worker.stop()
                clock.return_value = 23.0
                return b"wav", 3.0
            engine.synthesize.side_effect = synthesize
            worker.run()
        self.assertTrue(outgoing.empty())

    def test_overload_keeps_newest_in_order_and_join_works(self):
        q = LatestQueue(2, "test")
        for i in range(5):
            q.put_nowait(i)
        self.assertEqual(q.dropped, 3)
        self.assertEqual([q.get_nowait(), q.get_nowait()], [3, 4])
        q.task_done()
        q.task_done()
        self.assertEqual(q.unfinished_tasks, 0)
        q.join()

    def test_age_includes_time_waiting_for_stt(self):
        with patch("src.realtime.time.monotonic", return_value=20):
            self.assertTrue(expired({"segment_start": 7}, 12))
            self.assertFalse(expired({"segment_start": 9}, 12))

    def test_cuda_detection_does_not_depend_on_torch(self):
        ct2 = Mock()
        ct2.get_cuda_device_count.return_value = 1
        with patch.dict(sys.modules, {"ctranslate2": ct2}):
            backend = WhisperBackend(queue.Queue(), queue.Queue())
            self.assertEqual(backend._resolve_device(), ("cuda", "int8_float16"))
            ct2.get_cuda_device_count.return_value = 0
            self.assertEqual(backend._resolve_device(), ("cpu", "int8"))

    def test_explicit_cpu_never_uses_float16(self):
        backend = WhisperBackend(queue.Queue(), queue.Queue(), device="cpu")
        self.assertEqual(backend._resolve_device(), ("cpu", "int8"))

    def test_vad_emission_preserves_original_start(self):
        q = queue.Queue()
        vad = VADSegmenter(queue.Queue(), q)
        vad._model = Mock()
        vad._emit(np.zeros(16000, dtype=np.float32), 42.0)
        item = q.get_nowait()
        self.assertEqual(item["segment_start"], 42.0)
        self.assertEqual(item["duration"], 1.0)

    def test_vad_splits_continuous_speech_without_losing_samples(self):
        frames, segments = queue.Queue(), queue.Queue()
        audio = np.ones(512 * 10, dtype=np.float32)
        frames.put(audio)
        vad = VADSegmenter(frames, segments, max_segment_seconds=.128,
                           min_speech_ms=1, pre_roll_ms=0)
        vad._model = Mock()
        count = 0
        def probability(chunk):
            nonlocal count
            count += 1
            if count == 10:
                vad.stop()
            return 1.0
        vad._prob = probability
        vad.run()
        first, second = segments.get_nowait(), segments.get_nowait()
        self.assertEqual(len(first["audio"]) + len(second["audio"]), 512 * 8)
        self.assertLess(first["segment_start"], second["segment_start"])


if __name__ == "__main__":
    unittest.main()
