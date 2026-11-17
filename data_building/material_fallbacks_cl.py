"""
material_fallbacks_cl.py

Fallback acoustic materials for HM3D semantic categories.

RLR gives each mesh category the material with the most labels that are substrings
of the lower-cased category name. The mp3d material file only labels mpcat40-style
names, so about two thirds of the raw HM3D vocabulary ("couch", "lamp", "ceilling",
"kitchen island", ...) matches nothing and is simulated with the engine default.

write_scene_material_file() writes a per-scene copy of the material file in which every
category of the scene gets an explicit material. Everything else in this file is the
matching logic behind it; ACOUSTIC_FALLBACKS is the table to edit.
"""

import copy
import difflib
import json
import re

#Each entry maps a target to keywords found in HM3D names. A target is either a label
#already in the material file, so the category inherits that file's own choice
#("couch" -> "sofa" -> Curtain), or a material name for classes the file never labels
#(pictures, paper, outdoor ground, small objects). Keywords are matched as whole words
#(plurals and close typos included) with the head noun first; multi-word keywords are
#matched as phrases.
ACOUSTIC_FALLBACKS = {
    #structure
    "ceiling": "roof, soffit",
    "wall": "baseboard, skirting, molding, moulding, cornice, paneling, panelling, niche, "
            "relief, partition, parapet, column, pillar, arch, archway, ledge",
    "floor": "landing, platform, stage",
    "window": "skylight, stained glass",
    "door": "gate, hatch, doorway, entrance",
    "stairs": "step, ladder, riser",
    "railing": "balustrade, banister, bannister, handhold, rail, fence, grab bar, bath bar",
    "beam": "rafter, balk, joist, truss",
    "fireplace": "chimney, hearth, mantel, mantle, firebox, wood burner",
    #outdoor ground and openings
    "Concrete, Rough": "pavement, paving, patio, terrace, driveway, sidewalk, curb, "
                       "balcony, porch, deck, courtyard",
    "Grass": "lawn, grass, turf",
    "Gravel": "gravel",
    "Soil": "dirt, soil, ground, mud",
    "Water": "pool, pond, fountain",
    "Sound Proof": "sky",
    "plant": "tree, flower, bouquet, bush, shrub, hedge, vegetation, foliage, leaf, leaves, "
             "wreath, twig, branch, bonsai",
    #soft furnishings and textiles
    "cushion": "pillow, pouf, pouffe, poufe, ottoman, foam",
    "sofa": "couch, loveseat, futon, chaise",
    "bed": "mattress, crib, cradle, cot, bunk, headboard, hammock",
    "clothes": "coat, jacket, shirt, blouse, dress, robe, bathrobe, laundry, sock, hat, "
               "glove, shoe, boot, sandal, slipper, apron",
    "curtain": "drape, drapery, tapestry, banner, flag, canopy, tent, teepee, linen, sheet, "
               "duvet, quilt, doily, napkin, rag, handkerchief, tissue, fabric, textile, "
               "wool, fur, plush, stuffed, teddy, doll, puppet, lamp shade, lampshade",
    "Carpet": "carpet",
    "blinds": "blind, blinder, shade, shutter",
    #furniture
    "chest_of_drawers": "dresser, drawer, bureau, chest, commode",
    "cabinet": "closet, cupboard, pantry, hutch, sideboard, credenza, vitrine",
    "shelf": "shelve, bookcase, rack",
    "table": "stand, vanity, workstation",
    "counter": "island, worktop, bar, buffet",
    "seating": "bench, seat, lounger, recliner, backrest",
    "stool": "footrest, foot rest",
    "board_panel": "board, panel, plank, whiteboard, blackboard, noticeboard, corkboard, "
                   "dartboard",
    "furniture": "storage, unit, furnishing",
    "wood, Thick": "wood, wooden, timber, log, firewood, piano, sauna, stick",
    "wood, Thin": "picture, photo, photograph, photography, poster, painting, art, artwork, "
                  "canvas, portrait, image, drawing, frame, map, calendar, diploma, certificate, "
                  "sign, advertisement, menu, book, magazine, newspaper, paper, album, binder, "
                  "folder, notebook, document, file, brochure, leaflet, letter, card, cardboard, "
                  "box, crate, package, carton, basket, guitar, violin, ukulele",
    #hard, reflective
    "lighting": "lamp, light, chandelier, sconce, lantern, bulb, fixture",
    "tv_monitor": "tv, television, monitor, display, screen, laptop, computer, projector",
    "Glass": "glass, glazing, pane, bottle, jar, aquarium, fish tank",
    "appliances": "appliance, fridge, refrigerator, freezer, oven, stove, cooker, hob, stovetop, "
                  "dishwasher, washer, dryer, kettle, toaster, boiler, furnace, heater, radiator, "
                  "conditioner, dehumidifier, humidifier, purifier, fan, vent, ventilation, "
                  "ventilator, duct, hood, extractor, machine, maker, processor, compressor, "
                  "printer, copier, scanner, shredder, vacuum, mower, calculator, phone, "
                  "telephone, radio, stereo, player, soundbar, sound bar, speaker, subwoofer, "
                  "amplifier, amp, router, modem, icebox, electric, electrical, electronics, "
                  "device",
    "gym_equipment": "gym, exercise, fitness, workout, trainer, dumbbell, dumb bell, barbell, "
                     "weight, treadmill, trampoline",
    "Steel": "metal, steel, iron, aluminium, aluminum, chrome, brass, copper, pipe, piping, "
             "faucet, tap, valve, handle, knob, hook, rod, pole, chain, tank, can, tin, safe, "
             "locker, mailbox, bucket, pan, pot, cart, trolley, wheelbarrow, bicycle, bike, "
             "motorcycle, car, wheel, grate, grill, barbecue, extinguisher, trumpet, trombone, "
             "saxophone, clarinet, drum",
    "toilet": "urinal, bidet, potty, chamber pot",
    "sink": "basin, washbasin",
    "bathtub": "tub, jacuzzi, bath",
    "Tile, Ceramic": "ceramic, porcelain, tile, tiling, plate, dish, bowl, cup, mug, teapot, jug, "
                     "pitcher, vase, flowerpot, flower pot, platter, saucer, dinnerware, "
                     "backsplash, figurine",
    "Concrete": "stone, marble, granite, concrete, cement, stonework, rock, statue, sculpture, "
                "bust, slab",
    "Brick": "brick",
    "Acoustic Tile": "acoustic",
}

#words naming what something is made of win over the noun ("stone bench", "metal rack")
_MATERIAL_WORDS = {
    "glass", "metal", "steel", "iron", "aluminium", "aluminum", "chrome", "brass", "copper",
    "wood", "wooden", "timber", "stone", "marble", "granite", "concrete", "cement", "brick",
    "ceramic", "porcelain", "tile", "fabric", "textile", "wool", "fur", "foam", "acoustic",
}
_FALLBACK_KEYWORDS = {
    keyword.strip(): target
    for target, keywords in ACOUSTIC_FALLBACKS.items()
    for keyword in keywords.split(",")
}
#real words in the HM3D vocabulary that sit one typo away from a label or keyword
_NOT_TYPOS = {
    "shell", "paint", "tower", "plane", "self", "closing", "star", "wind", "brush",
    "marker", "title", "trailer",
}
#"box of tissues" / "basket /w books" / "flowers in vase": the head comes before these
_PHRASE_SPLIT = re.compile(r"\b(?:of|with|w|in|and|or|for|on)\b|[/,(]")



def write_scene_material_file(material_file: str, categories, out_path: str) -> tuple[int, int, int]:
    """Write a copy of material_file in which every category of a scene has a material.

    @param:
        material_file: the original RLR material file (e.g. mp3d_material_config.json)
        categories: the semantic category names present in the scene
        out_path: where the augmented copy is written
    @return: the counts returned by augment_material_labels
    """
    with open(material_file) as f:
        database = json.load(f)
    stats = augment_material_labels(database["materials"], categories)
    with open(out_path, "w") as f:
        json.dump(database, f)
    return stats


def augment_material_labels(materials: list[dict], categories) -> tuple[int, int, int]:
    """Give every category of a scene an explicit material.

    Categories no label matches are appended as labels to their fallback material; those
    that end on the default material are left to RLR's own default. Because RLR counts
    every label hit, a short added label can tip a longer category ("tin" inside
    "painting"), so any category whose intended material is no longer a strict winner
    gets another vote. Categories the original file already matched keep their material.
    @param:
        materials: the material list, modified in place
        categories: the category names present in the scene
    @return: (categories without an original match, of those left on the default,
    categories that still do not resolve to their intended material)
    """
    original = copy.deepcopy(materials)
    default = rlr_material_index("default", original)
    default = 0 if default is None else default

    enforced, num_fallback, num_default = {}, 0, 0
    for category in sorted({c.lower() for c in categories if c}):
        index = rlr_material_index(category, original)
        if index is None:
            num_fallback += 1
            index = fallback_material_index(category, original)
            if index == default:
                num_default += 1
                continue
            materials[index].setdefault("labels", []).append(category)
            enforced[category] = index
        elif _is_strict_winner(category, original, index):
            enforced[category] = index

    for _ in range(5):
        unstable = [c for c, i in enforced.items() if not _is_strict_winner(c, materials, i)]
        if not unstable:
            break
        for category in unstable:
            materials[enforced[category]]["labels"].append(category)
    unstable = [c for c, i in enforced.items() if not _is_strict_winner(c, materials, i)]

    return num_fallback, num_default, len(unstable)


def rlr_material_index(category: str, materials: list[dict]):
    """Mirror RLR's material lookup for one mesh category: the material whose labels
    occur most often as substrings of the lower-cased name (first one wins a tie).
    @return: the material index, or None when RLR would use its built-in default
    """
    hits = _label_hits(category, materials)
    best = max(hits, default=0)
    return hits.index(best) if best > 0 else None


def fallback_material_index(category: str, materials: list[dict]) -> int:
    """Choose a material for a category that none of the material file's labels match.

    Tried in order: spacing variants of a label ("chest of drawers" -> chest_of_drawers),
    multi-word keywords, material words, nouns head first ("picture frame" -> frame,
    "box of tissues" -> box), typo-tolerant matches against keywords and labels
    ("ceilling" -> ceiling), and finally the file's default material.
    @param:
        category: the HM3D category name
        materials: the material list as loaded from the material file (not augmented)
    @return: the material index
    """
    name = category.lower()
    names = {m.get("name"): i for i, m in enumerate(materials)}
    default = rlr_material_index("default", materials)
    default = 0 if default is None else default

    def resolve(target):
        if target in names:
            return names[target]
        index = rlr_material_index(target, materials)
        return default if index is None else index

    for variant in (name.replace(" ", "_"), name.replace(" ", "-")):
        index = rlr_material_index(variant, materials)
        if index is not None:
            return index

    def stems(w):
        return (w, w[:-1] if w.endswith("s") else w, w[:-2] if w.endswith("es") else w)

    def close_match(w, vocabulary, cutoff):
        #typos rarely change the first letter; requiring it keeps real words
        #from being "corrected" (heating -> seating)
        if w in _NOT_TYPOS:
            return None
        same_start = [v for v in vocabulary if v[0] == w[0]]
        return (difflib.get_close_matches(w, same_start, n=1, cutoff=cutoff) or [None])[0]

    #misspelt labels: treat "widnow frame" exactly like "window frame"
    labels = sorted({
        l.lower() for m in materials for l in m.get("labels", []) if l.isalpha() and len(l) >= 4
    })
    tokens = re.findall(r"[a-z]+", name)
    fixed = [
        (close_match(w, labels, 0.8) or w)
        if len(w) >= 4 and not any(f in _FALLBACK_KEYWORDS for f in stems(w)) else w
        for w in tokens
    ]
    if fixed != tokens:
        index = rlr_material_index(" ".join(fixed), materials)
        if index is not None:
            return index

    for keyword, target in _FALLBACK_KEYWORDS.items():
        if " " in keyword and keyword in name.replace("-", " "):
            return resolve(target)

    words = [
        w for part in _PHRASE_SPLIT.split(name) for w in reversed(re.findall(r"[a-z]+", part))
    ]
    for allowed in (_MATERIAL_WORDS, _FALLBACK_KEYWORDS):
        for w in words:
            for f in stems(w):
                if f in allowed and f in _FALLBACK_KEYWORDS:
                    return resolve(_FALLBACK_KEYWORDS[f])

    #misspelt keywords ("pilllow", "dishawasher"); stricter, words are real english
    keywords = sorted(k for k in _FALLBACK_KEYWORDS if " " not in k)
    for w in words:
        close = close_match(w, keywords, 0.85) if len(w) >= 5 else None
        if close:
            return resolve(_FALLBACK_KEYWORDS[close])

    return default


def _label_hits(category: str, materials: list[dict]) -> list[int]:
    category = category.lower()
    return [
        sum(1 for label in m.get("labels", []) if label and label.lower() in category)
        for m in materials
    ]


def _is_strict_winner(category: str, materials: list[dict], index: int) -> bool:
    hits = _label_hits(category, materials)
    return hits[index] > max((h for i, h in enumerate(hits) if i != index), default=0)
