#!/usr/bin/env python3
"""Test script for text overlay visual testing.

Generates test images with text overlays to verify:
- Font size is appropriate
- No text overflow
- Proper margins
- 2-line wrapping works correctly
- 4-5 words display properly
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def find_ffmpeg() -> str:
    """Find ffmpeg executable."""
    ffmpeg_paths = [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
        "/usr/bin/ffmpeg",
    ]
    for path in ffmpeg_paths:
        try:
            result = subprocess.run(
                [path, "-version"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            return path
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            continue
    raise RuntimeError("ffmpeg not found")

def find_font() -> str:
    """Find Poppins Bold font."""
    font_paths = [
        "/usr/share/fonts/truetype/poppins/Poppins-Bold.ttf",
        "/usr/share/fonts/truetype/Poppins-Bold.ttf",
        "/usr/local/share/fonts/Poppins-Bold.ttf",
        "~/.fonts/Poppins-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Poppins-Bold.ttf",
        "~/Library/Fonts/Poppins-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica-Bold.ttf",
    ]
    for fp in font_paths:
        expanded = Path(fp).expanduser()
        if expanded.exists():
            return str(expanded)
    return None

def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get image width and height using ffprobe."""
    # Find ffprobe
    ffprobe_paths = [
        "/opt/homebrew/bin/ffprobe",
        "/usr/local/bin/ffprobe",
        "ffprobe",
        "/usr/bin/ffprobe",
    ]
    ffprobe_cmd = None
    for path in ffprobe_paths:
        try:
            result = subprocess.run(
                [path, "-version"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            ffprobe_cmd = path
            break
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    if not ffprobe_cmd:
        return 1080, 1080  # Default
    
    try:
        probe_cmd = [
            ffprobe_cmd, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            image_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            dimensions = result.stdout.strip().split("x")
            if len(dimensions) == 2:
                return int(dimensions[0]), int(dimensions[1])
    except Exception as e:
        pass  # Silent fallback
    return 1080, 1080  # Default

def calculate_font_size_and_wrap(
    text: str,
    img_width: int,
    img_height: int,
    max_width_ratio: float = 0.70,  # Use 70% of width (15% margin on each side) - more conservative
    min_font_size: int = 36,
    max_font_size: int = 56,  # Reduced from 72 to prevent overflow
) -> Tuple[int, List[str], int]:
    """Calculate optimal font size and wrap text into 2 lines.
    
    Returns:
        (font_size, wrapped_lines, line_height)
    """
    words = text.split()
    
    # Wrap into 2 lines naturally
    if len(words) <= 2:
        wrapped_lines = [text]
    elif len(words) == 3:
        wrapped_lines = [' '.join(words[:2]), words[2]]
    elif len(words) == 4:
        wrapped_lines = [' '.join(words[:2]), ' '.join(words[2:])]
    elif len(words) == 5:
        wrapped_lines = [' '.join(words[:3]), ' '.join(words[3:])]
    else:
        mid = len(words) // 2
        wrapped_lines = [' '.join(words[:mid]), ' '.join(words[mid:])]
    
    # Calculate max text width (with margins) - more conservative
    max_text_width = int(img_width * max_width_ratio)
    longest_line = max(len(line) for line in wrapped_lines)
    
    # Estimate font size: 1 character ≈ 0.65 * fontsize pixels wide
    # More accurate estimate for bold fonts (Poppins Bold is wider)
    # Formula: fontsize ≈ max_text_width / (longest_line * 0.65)
    estimated_fontsize = int(max_text_width / (longest_line * 0.65))
    
    # Clamp to reasonable range
    fontsize = max(min_font_size, min(max_font_size, estimated_fontsize))
    
    # Line height (spacing between lines)
    line_height = int(fontsize * 1.35)
    
    return fontsize, wrapped_lines, line_height

def add_text_overlay(
    image_path: str,
    text: str,
    output_path: str,
    ffmpeg_cmd: str,
    font_path: str = None,
) -> bool:
    """Add text overlay to image with proper sizing and margins."""
    
    img_width, img_height = get_image_dimensions(image_path)
    
    # Calculate font size and wrapping
    fontsize, wrapped_lines, line_height = calculate_font_size_and_wrap(
        text, img_width, img_height
    )
    
    print(f"  Image: {img_width}x{img_height}")
    print(f"  Text: '{text}' ({len(text.split())} words)")
    print(f"  Wrapped: {wrapped_lines}")
    print(f"  Font size: {fontsize}px")
    print(f"  Max text width: {int(img_width * 0.70)}px (70% of image width)")
    print(f"  Longest line: {max(len(line) for line in wrapped_lines)} chars")
    
    # Escape text for FFmpeg
    wrapped_text = "\\n".join(wrapped_lines)
    escaped_wrapped = wrapped_text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace("[", "\\[").replace("]", "\\]")
    
    # Calculate positioning
    bottom_margin = max(80, int(img_height * 0.1))  # 10% margin from bottom, minimum 80px
    
    # Build FFmpeg drawtext filter
    drawtext_parts = [
        f"text='{escaped_wrapped}'",
        f"fontsize={fontsize}",
        "fontcolor=white",
        "borderw=2",
        "bordercolor=black@0.6",
        "x=(w-text_w)/2",  # Centered horizontally
        f"y=h-th-{bottom_margin}",  # Bottom with margin
        "shadowcolor=black@0.9",
        "shadowx=3",
        "shadowy=3",
    ]
    
    if font_path:
        drawtext_parts.append(f"fontfile={font_path}")
    
    drawtext_filter = "drawtext=" + ":".join(drawtext_parts)
    
    # Build FFmpeg command
    cmd = [
        ffmpeg_cmd,
        "-i", image_path,
        "-vf", drawtext_filter,
        "-y",
        output_path,
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0 and Path(output_path).exists():
            return True
        else:
            print(f"  ❌ FFmpeg error: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def create_test_images():
    """Create test images with various text overlays."""
    
    print("=" * 70)
    print("🧪 Text Overlay Visual Test Script")
    print("=" * 70)
    print()
    
    # Find ffmpeg and font
    try:
        ffmpeg_cmd = find_ffmpeg()
        print(f"✅ Found ffmpeg: {ffmpeg_cmd}")
    except RuntimeError:
        print("❌ ffmpeg not found. Please install it first.")
        return
    
    font_path = find_font()
    if font_path:
        print(f"✅ Found font: {font_path}")
    else:
        print("⚠️  Font not found, will use system default")
    
    print()
    
    # Test cases: (text, description)
    test_cases = [
        ("Refined skincare, not complicated", "4 words - should split 2-2"),
        ("Elevate your skin daily", "4 words - should split 2-2"),
        ("Clear skin, quiet confidence", "4 words - should split 2-2"),
        ("Purposeful living, daily care", "4 words - should split 2-2"),
        ("Routine refines your skin presence", "5 words - should split 3-2"),
        ("Discipline shows in skin care", "5 words - should split 3-2"),
        ("Consistent excellence builds confidence", "4 words - should split 2-2"),
        ("Quality skincare for men", "4 words - should split 2-2"),
    ]
    
    # Create test output directory
    test_dir = Path(__file__).parent / "test_overlay_output"
    test_dir.mkdir(exist_ok=True)
    
    # Check if we have a test image
    # Try to find an existing image or create a simple one
    test_image_path = None
    
    # Look for existing test images
    possible_test_images = [
        test_dir / "test_image.jpg",
        test_dir / "test_image.png",
        Path(__file__).parent / "test_image.jpg",
        Path(__file__).parent / "test_image.png",
    ]
    
    for img_path in possible_test_images:
        if img_path.exists():
            test_image_path = str(img_path)
            break
    
    # If no test image found, create a simple colored image
    if not test_image_path:
        test_image_path = str(test_dir / "test_base.png")
        print(f"📸 Creating test base image: {test_image_path}")
        
        # Create a 1080x1080 solid color image (simulating a photo background)
        create_cmd = [
            ffmpeg_cmd,
            "-f", "lavfi",
            "-i", f"color=c=0x2a2a2a:s=1080x1080:d=1",
            "-frames:v", "1",
            "-y",
            test_image_path,
        ]
        
        try:
            result = subprocess.run(create_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and Path(test_image_path).exists():
                print(f"✅ Created test base image")
            else:
                # Try alternative: use testsrc2 for a more interesting background
                create_cmd2 = [
                    ffmpeg_cmd,
                    "-f", "lavfi",
                    "-i", "testsrc2=size=1080x1080:duration=1",
                    "-frames:v", "1",
                    "-y",
                    test_image_path,
                ]
                result2 = subprocess.run(create_cmd2, capture_output=True, text=True, timeout=10)
                if result2.returncode == 0 and Path(test_image_path).exists():
                    print(f"✅ Created test base image (alternative method)")
                else:
                    print(f"❌ Failed to create test image")
                    print(f"   Error: {result2.stderr[:200]}")
                    print("   Please provide a test image manually at:")
                    print(f"   {test_image_path}")
                    return
        except Exception as e:
            print(f"❌ Failed to create test image: {e}")
            print(f"   Please provide a test image manually at:")
            print(f"   {test_image_path}")
            return
    
    print(f"📸 Using test image: {test_image_path}")
    print()
    
    # Process each test case
    print("Generating test images...")
    print("-" * 70)
    
    success_count = 0
    for i, (text, description) in enumerate(test_cases, 1):
        output_path = test_dir / f"test_{i:02d}_{text.replace(' ', '_').replace(',', '')[:30]}.png"
        
        print(f"\nTest {i}/{len(test_cases)}: {description}")
        print(f"  Text: '{text}'")
        
        success = add_text_overlay(
            test_image_path,
            text,
            str(output_path),
            ffmpeg_cmd,
            font_path,
        )
        
        if success:
            print(f"  ✅ Saved: {output_path}")
            success_count += 1
        else:
            print(f"  ❌ Failed")
    
    print()
    print("=" * 70)
    print(f"✅ Generated {success_count}/{len(test_cases)} test images")
    print(f"📁 Output directory: {test_dir}")
    print()
    print("💡 Review the images and adjust parameters in calculate_font_size_and_wrap()")
    print("   if needed:")
    print("   - max_width_ratio: Currently 0.70 (70% of width, 15% margin each side)")
    print("   - min_font_size: Currently 36px")
    print("   - max_font_size: Currently 56px")
    print("   - Character width estimate: Currently 0.65 * fontsize (for bold fonts)")
    print("=" * 70)

if __name__ == "__main__":
    create_test_images()

