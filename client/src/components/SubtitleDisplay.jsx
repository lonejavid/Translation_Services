import React, { useMemo, memo } from "react";

function pickCue(subtitles, currentTime) {
  if (!subtitles?.length) return null;
  const t = currentTime ?? 0;
  for (let i = subtitles.length - 1; i >= 0; i--) {
    const s = subtitles[i];
    if (t >= s.start && t <= s.end) return s;
  }
  return null;
}

function cueStableKey(s) {
  if (!s) return "";
  const o = s.original ?? s.text ?? "";
  const tr = s.translated ?? s.translated_text ?? s.text ?? "";
  return `${s.start}|${s.end}|${o}|${tr}`;
}

function SubtitleDisplayInner({ subtitles, currentTime }) {
  const current = useMemo(
    () => pickCue(subtitles, currentTime),
    [subtitles, currentTime]
  );

  if (!current) return null;

  const original = current.original ?? current.text ?? "";
  const translated =
    current.translated ?? current.translated_text ?? current.text ?? "";
  const showOriginal = Boolean(original && original !== translated);

  return (
    <div className="yt-sub" key={cueStableKey(current)}>
      <div className="yt-sub-lines">
        {showOriginal && <p className="yt-sub-original">{original}</p>}
        <p className="yt-sub-translated">{translated}</p>
      </div>
    </div>
  );
}

function subtitlePropsEqual(prev, next) {
  const a = pickCue(prev.subtitles, prev.currentTime);
  const b = pickCue(next.subtitles, next.currentTime);
  return cueStableKey(a) === cueStableKey(b);
}

export default memo(SubtitleDisplayInner, subtitlePropsEqual);
