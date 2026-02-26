import lzma
import urllib.request
from collections.abc import Iterator

from theseus.data.datasets import StreamingPretrainingDataset


_SENTENCE_URL_TEMPLATE = (
    "https://data.statmt.org/cc-aligned/sentence-aligned/en_XX-{lang}.tsv.xz"
)


class CCAligned(StreamingPretrainingDataset):
    """Multilingual parallel text from CCAligned.

    Streams sentence-aligned English <-> target language pairs directly
    from statmt.org, decompressing on-the-fly without downloading the
    whole file first.  The ``config`` parameter selects the target language
    code (e.g. ``"fr_XX"``, ``"de_DE"``, ``"zh_CN"``).  Each yielded
    string is the concatenation of the English and target sentences
    separated by a newline, suitable for multilingual pretraining.
    """

    def __init__(
        self,
        config: str | None = "fr_XX",
        split: str = "train",
    ) -> None:
        self.lang = config or "fr_XX"
        self._url = _SENTENCE_URL_TEMPLATE.format(lang=self.lang)

    def __iter__(self) -> Iterator[str]:
        response = urllib.request.urlopen(self._url)  # noqa: S310
        decompressor = lzma.LZMADecompressor()
        leftover = b""
        for chunk in iter(lambda: response.read(64 * 1024), b""):
            raw = decompressor.decompress(chunk)
            lines = (leftover + raw).split(b"\n")
            leftover = lines.pop()
            for line in lines:
                parts = line.decode("utf-8", errors="replace").split("\t")
                if len(parts) >= 2:
                    en_text = parts[0].strip()
                    tgt_text = parts[1].strip()
                    if en_text and tgt_text:
                        yield f"{en_text}\n{tgt_text}"
        # Handle final leftover
        if leftover:
            parts = leftover.decode("utf-8", errors="replace").split("\t")
            if len(parts) >= 2:
                en_text = parts[0].strip()
                tgt_text = parts[1].strip()
                if en_text and tgt_text:
                    yield f"{en_text}\n{tgt_text}"
        response.close()
