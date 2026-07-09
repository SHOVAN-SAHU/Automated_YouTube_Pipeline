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

# ── Negative prompt — compressed to save token budget for scene content ────────
NEGATIVE_BAN = (
    "photorealism, 3D render, realistic anatomy, extra/missing limbs, deformed hands, "
    "distorted face, modern clothing/objects, text, watermark, blurry, grayscale, "
    "sepia, gradient shading, airbrushed, glossy, cropped character"
)

# ── Shot type rotation — cycles through these to create visual variety ─────────
# Each scene gets assigned a shot type based on its sequence number, creating
# a natural rhythm of wide→medium→close-up→POV→etc. instead of the same
# center-framed medium shot every time.
SHOT_TYPES = [
    "extreme wide establishing shot — character small in a vast landscape, emphasizing environment",
    "medium shot — character from waist up, balanced with background",
    "close-up — character's face and upper body fill most of the frame, background blurred/minimal",
    "wide shot — full character visible with generous background context",
    "low-angle shot looking up — character appears powerful/imposing, sky visible above",
    "bird's-eye view — looking straight down at character and surroundings from above",
    "over-shoulder shot — camera behind character looking at what they see ahead",
    "medium-wide shot — character at one-third of frame, environment fills the rest",
    "extreme close-up — just the character's face/hands and one key object, very tight crop",
    "panoramic wide shot — ultra-wide landscape with character as a small figure in it",
]

# ── Composition directives — extra framing instructions per shot type ──────────
COMPOSITION_DIRECTIVES = {
    "extreme wide establishing shot": "Place character small (under 20% of frame) off-center. The environment IS the subject.",
    "medium shot": "Character at center or rule-of-thirds. Show waist-up with clear background.",
    "close-up": "Fill 60%+ of frame with character's head/shoulders. Background is simple, flat color or out-of-focus shapes.",
    "wide shot": "Full body visible. Character occupies ~30% of frame. Rich, detailed background.",
    "low-angle shot looking up": "Camera below character, looking up. Exaggerate character height. Sky/ceiling visible.",
    "bird's-eye view": "Top-down camera. Character seen from directly above. Ground/terrain fills frame.",
    "over-shoulder shot": "Camera positioned behind character's shoulder. Character's back visible. Focus on what's ahead.",
    "medium-wide shot": "Character at left or right third. Large environment element on the opposite side.",
    "extreme close-up": "Tight on face or hands + one object. No full body. Minimal background.",
    "panoramic wide shot": "Ultra-wide aspect emphasis. Character tiny, landscape dominant. Horizon line prominent.",
}

# ── Image model ─────────────────────────────────────────────────────────
# phoenix-1.0 is tuned for photoreal/painterly prompt-adherence, which is a poor
# match for holding one locked, simple flat-vector character across hundreds of
# independent calls. lucid-origin is Leonardo's model tuned specifically for
# "sharp graphic design... highly specific creative direction" and stylized
# looks — closer to what a flat minimalist explainer character needs.
# If it still hallucinates, flux-1-schnell is worth an A/B (faster, but weaker
# instruction-following on complex multi-clause prompts).
RECOMMENDED_IMAGE_MODEL = "@cf/leonardo/lucid-origin"