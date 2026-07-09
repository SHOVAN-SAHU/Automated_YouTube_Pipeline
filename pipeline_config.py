"""
Shared, locked visual identity for the pipeline.

Both generate_story.py and generate_images.py import from here. This is the
single source of truth for what the character/art style looks like — it used
to be re-described (loosely, by an LLM) in the story brief, then re-described
again (differently, hardcoded) in the image script. That drift between the
two is a big part of why renders didn't match: nothing was actually fixed.
"""

# ── The character (matches the reference screenshot / style doc) ──────────
CHARACTER_DESIGN = (
    "THE CHARACTER — one single consistent design, used in every image: a minimalist "
    "figure with a perfectly round, blank cream/white head, two small simple black dot "
    "eyes, NO nose, and a tiny simple mouth (a curved smile, a flat neutral line, or a "
    "worried frown — expression comes ONLY from eyebrows and mouth shape, the head "
    "itself stays blank and featureless otherwise). Thin, simple black-outlined stick "
    "limbs and a small simple torso. ANCIENT dress: a messy tousled dark hair tuft on "
    "top of the head, wrapped in a rough animal-hide/fur tunic with jagged uneven "
    "edges in warm spotted brown/orange tones, barefoot — no shoes, no modern clothing, "
    "no modern objects anywhere in frame. The character must look IDENTICAL across "
    "every single image: same head size/shape, same dot eyes, same tiny mouth, same "
    "limb thickness, same tunic. Only pose, expression, and camera angle change."
)

# ── Base art style (time-of-day / mood logic stays dynamic per scene) ─────
AESTHETIC_ANCHOR = (
    "Flat 2D cartoon explainer style: thick, uniform-width black outlines around every "
    "shape, solid FLAT color fills with no gradients or airbrushing (a touch of "
    "hand-drawn texture/shading is allowed only in extreme close-ups or harsh outdoor "
    "scenes). Backgrounds are simple and readable — a few clear shapes (rock walls, "
    "trees, sky, ground) rather than busy or photorealistic detail. Clean widescreen "
    "composition. Color and lighting follow the ACTUAL time of day/setting of the "
    "scene: bright cheerful light-blue sky and warm greens/browns for daylight or "
    "normal moments; darker, muddier browns/greys/deep greens with moody shadow for "
    "harsh survival/danger moments; plain cream/off-white background with no scenery "
    "for diagram or explainer beats. Never photorealistic, never 3D-rendered, never "
    "sepia or desaturated gray."
)

# ── Negative prompt — expanded to specifically fight the failure modes you saw ──
NEGATIVE_BAN = (
    "photorealism, 3D render, realistic human anatomy, extra limbs, extra fingers, "
    "missing limbs, deformed hands, distorted face, asymmetrical body, disproportionate "
    "head, character not matching described design, inconsistent character design, "
    "modern clothing, modern objects, cars, buildings, text, watermark, signature, "
    "blurry, low quality, grayscale, black and white photo, sepia, washed out colors, "
    "muddy colors, gradient shading, airbrushed, glossy render, cropped character, "
    "character cut off at edge of frame"
)

# ── Image model ─────────────────────────────────────────────────────────
# phoenix-1.0 is tuned for photoreal/painterly prompt-adherence, which is a poor
# match for holding one locked, simple flat-vector character across hundreds of
# independent calls. lucid-origin is Leonardo's model tuned specifically for
# "sharp graphic design... highly specific creative direction" and stylized
# looks — closer to what a flat minimalist explainer character needs.
# If it still hallucinates, flux-1-schnell is worth an A/B (faster, but weaker
# instruction-following on complex multi-clause prompts).
RECOMMENDED_IMAGE_MODEL = "@cf/leonardo/lucid-origin"