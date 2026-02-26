import csv
import lzma
import tempfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from theseus.data.datasets import StreamingPretrainingDataset


_SENTENCE_URL_TEMPLATE = (
    "https://data.statmt.org/cc-aligned/sentence-aligned/"
    "en_XX-{lang}.tsv.xz"
)


class CCAligned(StreamingPretrainingDataset):
    """Multilingual parallel text from CCAligned.

    Streams sentence-aligned English <-> target language pairs downloaded
    from statmt.org.  The ``config`` parameter selects the target language
    code (e.g. ``"fr_XX"``, ``"de_DE"``, ``"zh_CN"``).  Each yielded
    string is the concatenation of the English and target sentences
    separated by a newline, suitable for multilingual pretraining.
    """

    def __init__(
        self,
        config: str | None = "fr_XX",
        split: str = "train",
    ) -> None:
        lang = config or "fr_XX"
        self.lang = lang

        url = _SENTENCE_URL_TEMPLATE.format(lang=lang)

        cache_dir = Path(tempfile.gettempdir()) / "theseus_ccaligned"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = cache_dir / f"en_XX-{lang}.tsv"

        if not self._cache_file.exists():
            xz_file = cache_dir / f"en_XX-{lang}.tsv.xz"
            urllib.request.urlretrieve(url, xz_file)  # noqa: S310
            with lzma.open(xz_file, "rb") as f_in:
                with open(self._cache_file, "wb") as f_out:
                    while True:
                        chunk = f_in.read(1024 * 1024)
                        if not chunk:
                            break
                        f_out.write(chunk)
            xz_file.unlink(missing_ok=True)

    def __iter__(self) -> Iterator[str]:
        with open(self._cache_file, encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    en_text = row[0].strip()
                    tgt_text = row[1].strip()
                    if en_text and tgt_text:
                        yield f"{en_text}\n{tgt_text}"
