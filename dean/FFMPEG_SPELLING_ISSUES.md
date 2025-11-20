# FFmpeg-Specific Causes of Spelling Mistakes

## 1. **FFmpeg Text Escaping Issues**
- Double backslashes `\\n` being interpreted as single `\n` by shell
- Single quotes in text parameter causing FFmpeg to break
- Escaping removing characters incorrectly
- Backslash escaping order issues

## 2. **FFmpeg drawtext Filter Text Parameter**
- FFmpeg interpreting text as expression instead of literal
- Text parameter requiring different escaping than expected
- FFmpeg truncating text at certain characters
- Special characters in filter syntax being interpreted

## 3. **FFmpeg Line Break Handling**
- `\n` vs `\\n` escaping issues
- FFmpeg not recognizing line breaks correctly
- Line spacing parameter interfering with text
- Multiline text rendering incorrectly

## 4. **FFmpeg Command Line Parsing**
- Shell interpreting quotes before FFmpeg sees them
- Command line argument parsing issues
- Filter string being split incorrectly
- Special characters in filter causing parsing errors

## 5. **FFmpeg Font Rendering**
- Font file not being found, using fallback
- Font fallback causing character substitution
- Font metrics calculation issues
- Character width estimation wrong for bold fonts

## 6. **FFmpeg Text Expression Mode**
- FFmpeg treating text as expression (evaluating variables)
- Need to escape `:` or `=` in text
- Text containing filter syntax characters
- FFmpeg version differences in text handling

## 7. **FFmpeg Character Encoding**
- FFmpeg not handling UTF-8 correctly
- Character encoding mismatch
- Special characters being corrupted
- Font encoding issues

## 8. **FFmpeg Filter String Construction**
- Filter string too long being truncated
- Quotes in filter string breaking parsing
- Colon `:` in text being interpreted as parameter separator
- Equal sign `=` in text being interpreted as assignment

## 9. **FFmpeg Version/Platform Differences**
- Different FFmpeg versions handle text differently
- libav vs ffmpeg differences
- Platform-specific text rendering
- Different text rendering engines

## 10. **FFmpeg Text Parameter Escaping Rules**
- Need to escape: `\`, `'`, `:`, `[`, `]`, `=`
- Escaping order matters
- Double escaping issues
- Shell vs FFmpeg escaping conflicts

