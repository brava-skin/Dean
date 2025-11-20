#!/usr/bin/env python3
"""
Test script for text overlay generation and spacing fixes.
Tests the _fix_text_spacing_errors function with various inputs.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from creative.image_generator import ImageCreativeGenerator

def test_text_spacing_fixes():
    """Test the text spacing fix function with known problematic inputs."""
    
    # Use the actual implementation instead of a mock
    # Create a minimal generator instance
    try:
        # Try to create with None clients (will fail, but we can work around it)
        generator = ImageCreativeGenerator.__new__(ImageCreativeGenerator)
        # Set minimal attributes needed
        generator.flux_client = None
    except Exception as e:
        print(f"Warning: Could not create generator: {e}")
        # Fallback: import and call the function directly
        from creative.image_generator import ImageCreativeGenerator
        # Create a dummy instance
        class DummyFlux:
            pass
        generator = ImageCreativeGenerator.__new__(ImageCreativeGenerator)
        generator.flux_client = None
    
    # Test cases: (input, expected_output)
    # Testing both known issues and NEW merged words (not in patterns)
    test_cases = [
                # Known issues
                ("quietnpresence", "quiet presence"),
                ("refined skin, quietnpresence.", "refined skin, quiet presence."),
                ("withnskin", "with skin"),
                ("innskin", "in skin"),
                ("yournskin", "your skin"),
                ("beginswith", "begins with"),
                ("showsin", "shows in"),
                ("refinesyour", "refines your"),
                ("respectshows", "respect shows"),
                # NEW merged words (not in patterns - testing general approach)
                ("calmconfidence", "calm confidence"),  # New merge
                ("dailyroutine", "daily routine"),  # New merge
                ("premiumquality", "premium quality"),  # New merge
                ("strengthshows", "strength shows"),  # New merge
                # Valid words that should NOT be split
                ("purposeful", "purposeful"),  # Valid compound word - should NOT be split
                ("Purposeful living.", "Purposeful living."),  # Valid compound word - should NOT be split
                ("Purpose ful living.", "Purpose ful living."),  # Note: This might not fix "Purpose ful" -> "Purposeful" (that's a different issue)
                ("skincare", "skincare"),  # Valid compound word - should NOT be split
                ("long term", "long term"),
                ("presence", "presence"),
                ("authority", "authority"),
                ("elevate your presence", "elevate your presence"),
                ("calm authority", "calm authority"),
                ("Built for the long term.", "Built for the long term."),
                ("confidence", "confidence"),  # Single valid word
                ("routine", "routine"),  # Single valid word
            ]
    
    print("🧪 Testing text spacing fixes...\n")
    
    all_passed = True
    for input_text, expected in test_cases:
        result = generator._fix_text_spacing_errors(input_text)
        passed = result == expected
        status = "✅" if passed else "❌"
        
        if not passed:
            all_passed = False
            print(f"{status} FAILED:")
            print(f"   Input:    '{input_text}'")
            print(f"   Expected: '{expected}'")
            print(f"   Got:      '{result}'")
        else:
            print(f"{status} '{input_text}' -> '{result}'")
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("="*60)
    
    return all_passed


def test_full_overlay_generation():
    """Test full text overlay generation with a sample image."""
    
    print("\n🖼️  Testing full text overlay generation...\n")
    
    # Check if we have required environment variables
    flux_api_key = os.getenv("FLUX_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not flux_api_key:
        print("⚠️  FLUX_API_KEY not set - skipping full overlay test")
        return False
    
    if not openai_api_key:
        print("⚠️  OPENAI_API_KEY not set - skipping full overlay test")
        return False
    
    try:
        from integrations.flux_client import create_flux_client
        
        flux_client = create_flux_client()
        generator = ImageCreativeGenerator(
            flux_client=flux_client,
            openai_api_key=openai_api_key
        )
        
        # Test with problematic text
        test_text = "refined skin, quietnpresence."
        print(f"Testing with text: '{test_text}'")
        
        # Generate a simple test image
        print("Generating test image...")
        product_info = {
            "name": "Test Product",
            "description": "A premium men's skincare product for testing text overlays."
        }
        
        creative_data = generator.generate_creative(
            product_info=product_info,
            aspect_ratios=["1:1"]
        )
        
        if creative_data and "images_by_aspect" in creative_data:
            image_path = creative_data["images_by_aspect"].get("1:1")
            if image_path:
                print(f"✅ Image generated: {image_path}")
                
                # Test text overlay
                print(f"Adding text overlay: '{test_text}'")
                overlay_path = generator._add_text_overlay(image_path, test_text)
                
                if overlay_path:
                    print(f"✅ Text overlay created: {overlay_path}")
                    print(f"\n📝 Check the image at: {overlay_path}")
                    print(f"   Expected text: 'refined skin, quiet presence.'")
                    print(f"   Verify the spacing is correct in the image.")
                    return True
                else:
                    print("❌ Failed to create text overlay")
                    return False
            else:
                print("❌ No image path in creative data")
                return False
        else:
            print("❌ Failed to generate creative")
            return False
            
    except Exception as e:
        print(f"❌ Error in full overlay test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("🧪 Text Overlay Test Script")
    print("="*60)
    
    # Test 1: Text spacing fixes
    spacing_ok = test_text_spacing_fixes()
    
    # Test 2: Full overlay generation (optional, requires API keys)
    if spacing_ok:
        print("\n" + "="*60)
        print("💡 Tip: Set FLUX_API_KEY and OPENAI_API_KEY to test full overlay generation")
        print("="*60)
        
        # Only run full test if user wants (requires API calls)
        if len(sys.argv) > 1 and sys.argv[1] == "--full":
            overlay_ok = test_full_overlay_generation()
            sys.exit(0 if (spacing_ok and overlay_ok) else 1)
    else:
        print("\n⚠️  Fix spacing issues before testing full overlay generation")
        sys.exit(1)
    
    sys.exit(0 if spacing_ok else 1)

