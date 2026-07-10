"""
Shared, locked visual identity for the pipeline.

Both generate_story.py and generate_images.py import from here. This is the
single source of truth for what the character/art style looks like — it used
to be re-described (loosely, by an LLM) in the story brief, then re-described
again (differently, hardcoded) in the image script. That drift between the
two is a big part of why renders didn't match: nothing was actually fixed.
"""

# ── The character (matches the reference screenshot / style doc) ──────────
# NOTE: this text is a per-scene rendering instruction, not just internal
# documentation — it gets sent to the image model on every single call. Avoid
# any phrasing like "identical across every image" here: a model reads that as
# "show me multiple versions side by side" and renders a comparison grid
# instead of one scene (this is exactly what caused the split-panel renders).
CHARACTER_DESIGN = (
    "THE CHARACTER: a minimalist figure with a perfectly round, blank cream/white head, "
    "two small simple black dot eyes, NO nose, and a tiny simple mouth (a curved smile, a "
    "flat neutral line, or a worried frown — expression comes ONLY from eyebrows and mouth "
    "shape, the head itself stays blank and featureless otherwise). Thin, simple "
    "black-outlined stick limbs and a small simple torso. ANCIENT dress: a messy tousled "
    "dark hair tuft on top of the head, wrapped in a rough animal-hide/fur tunic with "
    "jagged uneven edges in warm spotted brown/orange tones, barefoot — no shoes, no "
    "modern clothing, no modern objects anywhere in frame."
)

# ── Hard guard against the grid/split-panel failure mode ──────────────────
SINGLE_PANEL_GUARD = (
    "A single full-bleed cinematic illustration filling the entire widescreen frame with "
    "one continuous, richly detailed background environment appropriate to the scene — the "
    "character is captured mid-moment, interacting with and surrounded by that environment "
    "(rocks, terrain, sky, structures, or other scenery filling the space around them)."
)

# ── Camera / shot variety — pick ONE per scene, rotate through these ──────
# This list existed in the original style doc but never made it into the actual
# render pipeline, which is a big part of why every scene came out as the same
# centered, medium-wide, standing shot regardless of what the line said.
CAMERA_SHOTS = [
    "wide establishing shot, character small within a large environment",
    "medium full-body shot, character centered",
    "extreme close-up on the face, filling most of the frame",
    "side-profile shot, character facing across the frame",
    "low angle looking up at the character, sky or cliffs looming above",
    "high angle / bird's-eye aerial shot, character tiny and isolated below",
    "over-the-shoulder or from-behind shot, character looking out into the scene",
    "symmetrical centered framing emphasizing smallness or vastness",
    "tight shot on hands/feet/a specific detail with the character partly out of frame",
]

# ── Base art style — deliberately has NO day/night/diagram branching text ──
# The old version listed three different lighting/background scenarios
# ("...for daylight moments; ...for harsh moments; ...for diagram beats") in
# one prompt. lucid-origin has no way to know those are alternative options to
# choose between — it just tries to satisfy all of them, which is exactly what
# produced the 2-panel and 3-panel split/comparison renders. The actual mood
# for THIS scene is decided once, per-scene, by the enhancer (see rule 5 in
# _build_enhance_prompt) and lands inside enhanced_prompt itself — this anchor
# only needs to state style rules that are true for every single image.
AESTHETIC_ANCHOR = (
    "Flat 2D hand-drawn cartoon explainer style: thick, uniform-width black outlines around "
    "every shape, solid FLAT color fills with no gradients or airbrushing (a touch of "
    "hand-drawn texture/shading is allowed only in extreme close-ups or harsh outdoor "
    "scenes). Backgrounds are simple and readable — a few clear shapes (rock walls, "
    "trees, sky, ground) rendered in the same flat vector-cartoon style as the character. "
    "Clean widescreen composition, colors fully saturated and vivid. " + SINGLE_PANEL_GUARD
)

# ── Negative guidance — for the TEXT enhancer LLM only, never the image model ──
# lucid-origin's API has no `negative_prompt` field (checked against Cloudflare's
# published schema — prompt/guidance/seed/width/height/steps only). Feeding this
# list into the image prompt itself as "Avoid: X, Y, Z" doesn't suppress those
# things — with no true negation, a diffusion model often half-renders the
# literal words instead (this is almost certainly why one render came out
# partially desaturated/grayscale — "grayscale, sepia" was sitting in the
# positive prompt). Use this ONLY when prompting a text-generating LLM (which
# does understand negation), to steer it away from describing these things in
# the first place — never concatenate it into the final image-render prompt.
NEGATIVE_BAN = (
    "photorealism, 3D render, realistic human anatomy, extra limbs, extra fingers, "
    "missing limbs, deformed hands, distorted face, asymmetrical body, disproportionate "
    "head, character not matching described design, inconsistent character design, "
    "modern clothing, modern objects, cars, buildings, text, watermark, signature, "
    "blurry, low quality, grayscale, sepia, washed out colors, muddy colors, gradient "
    "shading, airbrushed, glossy render, cropped character, character cut off at edge "
    "of frame, split screen, grid layout, multiple panels, comparison image"
)

# ── Per-scene mood snippets — the enhancer picks exactly ONE per scene ────
# These used to live combined inside AESTHETIC_ANCHOR as a single "for X...for
# Y...for Z" menu that got sent whole to every render. Now each is separate;
# the enhancer resolves the scene's actual lighting tag into ONE of these
# phrasings and writes that single choice into its own enhanced_prompt text.
MOOD_DAYLIGHT = "Bright cheerful light-blue sky, warm sunny greens and browns, fully lit daylight scene."
MOOD_HARSH = "Darker, muddier browns/greys/deep greens, moody shadow, harsh survival/danger atmosphere."
MOOD_NIGHT = "Deep blue night palette, warm orange firelight or cool blue moonlight accents."
MOOD_DIAGRAM = "Plain cream/off-white background, no scenery, flat explainer/diagram framing."

# ── Image model ─────────────────────────────────────────────────────────
# phoenix-1.0 is tuned for photoreal/painterly prompt-adherence, which is a poor
# match for holding one locked, simple flat-vector character across hundreds of
# independent calls. lucid-origin is Leonardo's model tuned specifically for
# "sharp graphic design... highly specific creative direction" and stylized
# looks — closer to what a flat minimalist explainer character needs.
# If it still hallucinates, flux-1-schnell is worth an A/B (faster, but weaker
# instruction-following on complex multi-clause prompts).
RECOMMENDED_IMAGE_MODEL = "@cf/leonardo/lucid-origin"