# Running Dean ASC+ Campaign Locally

## Quick Start

1. **Make sure you have a `.env` file** in the `dean/` directory with all required variables:
   - `FB_APP_ID`, `FB_APP_SECRET`, `FB_ACCESS_TOKEN`
   - `FB_AD_ACCOUNT_ID`, `FB_PAGE_ID`
   - `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`
   - `OPENAI_API_KEY`, `FLUX_API_KEY`
   - `SLACK_WEBHOOK_URL` (optional)

2. **Run the debug script:**
   ```bash
   cd dean
   python3 scripts/run_asc_plus_debug.py
   ```

3. **Or run main.py directly:**
   ```bash
   cd dean
   python3 -u src/main.py
   ```

## What to Look For

The script will show detailed logs including:
- ✅ Creative generation (Flux images)
- ✅ Text overlay addition
- ✅ Supabase Storage uploads
- ✅ Meta API calls (create_image_creative, create_ad)
- ❌ Any errors preventing ads from being created

## Key Debug Points

Watch for these log messages:
- `"Processing creative #X: has supabase_storage_url=..."`
- `"Creating Meta creative and ad for creative #X..."`
- `"Creating Meta creative: name=..."`
- `"Creating ad with name=..."`
- `"✅ Successfully created creative #X: creative_id=..., ad_id=..."`

If you see creatives being generated but no "Successfully created" messages, check for:
- Meta API errors
- Missing environment variables
- Creative validation failures

