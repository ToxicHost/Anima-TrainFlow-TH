# Bundled fonts (offline)

Studio Trainer renders fully offline — no Google Fonts `<link>`, no CDN. The CSS
(`assets/app.css`) declares `@font-face` against the files below. Until they are
present the UI falls back to the system sans / mono (it still works and looks
correct, just not the exact Studio typefaces).

Drop these **woff2** files here (exact filenames matter — they're referenced in
`app.css`):

| File | Family / weight | Source |
|------|-----------------|--------|
| `DMSans-Light.woff2`        | DM Sans 300 | Google Fonts (OFL) |
| `DMSans-Regular.woff2`      | DM Sans 400 | Google Fonts (OFL) |
| `DMSans-Medium.woff2`       | DM Sans 500 | Google Fonts (OFL) |
| `DMSans-SemiBold.woff2`     | DM Sans 600 | Google Fonts (OFL) |
| `JetBrainsMono-Regular.woff2` | JetBrains Mono 400 | JetBrains (OFL) |
| `JetBrainsMono-Medium.woff2`  | JetBrains Mono 500 | JetBrains (OFL) |

Both families are SIL Open Font License, redistributable. Convert TTF → woff2
with any tool (e.g. `fonttools ttLib.woff2`) if you only have the TTFs.
