# Policy + rate-limit behavior

1. Fetch `/robots.txt` per origin.
2. `Disallow` => never crawl that URL.
3. `Crawl-delay` => effective delay = max(profile delay, robots delay).
4. robots unavailable => block by default.
5. 429 / 503 => Retry-After when present, otherwise exponential backoff.
6. 401 / 403 => explicit host block; stop host.
7. CAPTCHA / challenge text => stop host.
8. redirects are normalized + allowed-domain checked before being queued.
9. no auth/login/cookie bypass, no CAPTCHA bypass.
10. each member records site-specific policy research in `NOTES.md`.
