/**
 * Map YouTube timeline (seconds) → dubbed MP3 position (seconds).
 * When Hindi TTS is longer than the subtitle window, the audio file is laid out
 * sequentially; naive audio.currentTime = videoTime skips the tail of phrases.
 *
 * dub_sync entries (from server): video_start, video_end, audio_start, audio_end
 */
export function videoTimeToDubAudioTime(vt, map) {
  if (map == null || map.length === 0 || vt == null || Number.isNaN(vt)) {
    return vt;
  }
  const t = Math.max(0, vt);
  const first = map[0];
  if (t <= first.video_start) {
    return Math.min(t, Math.max(0, first.audio_start));
  }

  let lastAudioEnd = first.audio_start;
  for (const s of map) {
    const v0 = s.video_start;
    const v1 = s.video_end;
    const a0 = s.audio_start;
    const a1 = s.audio_end;
    const alen = Math.max(0, a1 - a0);
    const slot = Math.max(0, v1 - v0);
    const tail = Math.max(0, alen - slot);
    const tailEnd = v1 + tail;

    if (t < v0) {
      return lastAudioEnd;
    }
    if (t <= tailEnd) {
      const inV = Math.min(t, v1) - v0;
      const afterV = Math.max(0, t - v1);
      const playhead = Math.min(alen, inV + afterV);
      return a0 + playhead;
    }
    lastAudioEnd = a1;
  }
  return map[map.length - 1].audio_end;
}
