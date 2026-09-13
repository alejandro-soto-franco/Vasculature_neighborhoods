"""Download one file from Dryad and verify its SHA-256.

datadryad.org sits behind an Anubis (Techaro) proof-of-work anti-scraper
challenge. This solves it the same way a browser's JS would (the algorithm is
open and documented: sha256(random_data + nonce) with N leading zero hex
digits) rather than requiring a human to click through, then downloads the
file the challenge redirects to.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vasculature_neighborhoods.io import verify_sha256  # noqa: E402

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
# (connect timeout, read timeout): a stalled connection (the remote closes
# without either side noticing, observed once on this Dryad host under load)
# raises after this many idle seconds between chunks, rather than hanging
# indefinitely, so the retry loop below actually gets a turn.
_TIMEOUT = (15, 60)
_MAX_ATTEMPTS = 20


def _extract_challenge(html: str) -> dict:
    idx = html.find("anubis_challenge")
    start = html.find("{", idx)
    depth = 0
    end = None
    for i, c in enumerate(html[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return json.loads(html[start:end])


def _solve(random_data: str, difficulty: int) -> tuple[str, int]:
    half = difficulty // 2
    odd = difficulty % 2 != 0
    nonce = 0
    while True:
        digest = hashlib.sha256((random_data + str(nonce)).encode()).digest()
        ok = all(b == 0 for b in digest[:half])
        if ok and odd and (digest[half] >> 4) != 0:
            ok = False
        if ok:
            return digest.hex(), nonce
        nonce += 1


def _stream_to_file(resp: requests.Response, dest: str, resume_from: int) -> None:
    """Write a response body to ``dest``, appending if the server honoured a Range request."""
    mode = "ab" if resp.status_code == 206 else "wb"
    if mode == "wb" and resume_from:
        # Asked to resume (Range: bytes=N-) but the server sent a fresh 200
        # instead of 206: it does not support ranged requests here, so the
        # only correct thing is to restart from byte 0.
        resume_from = 0
    with open(dest, mode) as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)


def _resolve_data_url(session: requests.Session, url: str) -> str:
    """Solve the Anubis challenge (if shown) and return the final data URL.

    Does not download the body; used so a resume attempt can issue its own
    ranged GET directly against the resolved (presigned) URL rather than
    replaying the one-shot challenge, which cannot be redeemed twice.
    """
    resp = session.get(url, allow_redirects=True, timeout=_TIMEOUT, stream=True)
    content_type = resp.headers.get("content-type", "")
    content_length = int(resp.headers.get("content-length", "0") or 0)
    if not content_type.startswith("text/html") or content_length > 1_000_000:
        resp.close()
        return resp.url

    challenge = _extract_challenge(resp.text)
    resp.close()
    rules, ch = challenge["rules"], challenge["challenge"]
    t0 = time.time()
    hex_hash, nonce = _solve(ch["randomData"], rules["difficulty"])
    elapsed_ms = int((time.time() - t0) * 1000)

    pass_url = "https://datadryad.org/.within.website/x/cmd/anubis/api/pass-challenge"
    params = {
        "id": ch["id"],
        "response": hex_hash,
        "nonce": str(nonce),
        "redir": url,
        "elapsedTime": str(elapsed_ms),
    }
    # `stream=True` here is load-bearing, not an optimisation: without it,
    # `requests` reads the ENTIRE response body (the redirect chain ends at
    # the actual multi-GB S3 object) into memory before this call returns,
    # even though only `.url` is wanted. That is exactly the bug this
    # function exists to avoid repeating on every retry.
    resp2 = session.get(
        pass_url, params=params, allow_redirects=True, timeout=_TIMEOUT, stream=True
    )
    content_type = resp2.headers.get("content-type", "")
    if "text/html" in content_type:
        raise RuntimeError(f"anti-scraper challenge not resolved for {url}: {resp2.text[:500]}")
    resolved_url = resp2.url
    resp2.close()
    return resolved_url


def download(url: str, dest: str) -> None:
    """Download ``url`` to ``dest``, resuming a stalled or dropped connection.

    The anti-scraper challenge is one-shot, so it is solved once to resolve
    the real (presigned) data URL; every retry after that issues a ranged
    GET (``Range: bytes=<partial size>-``) directly against that resolved
    URL, appending rather than restarting, unless the server ignores the
    Range header (some proxies do), in which case it restarts from 0.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": _UA})
    data_url: str | None = None
    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            if data_url is None:
                data_url = _resolve_data_url(session, url)
            resume_from = Path(dest).stat().st_size if Path(dest).exists() else 0
            headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
            resp = session.get(
                data_url, headers=headers, timeout=_TIMEOUT, stream=True, allow_redirects=True
            )
            if resp.status_code not in (200, 206):
                resp.raise_for_status()
            _stream_to_file(resp, dest, resume_from)
            return
        except (requests.exceptions.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt < _MAX_ATTEMPTS:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(
        f"download failed after {_MAX_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def main() -> None:
    url, dest, expected_sha256 = sys.argv[1], sys.argv[2], sys.argv[3]
    download(url, dest)
    verify_sha256(dest, expected_sha256)


if __name__ == "__main__":
    main()
