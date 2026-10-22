"""Coarse taxon groups used to constrain BioCLIP species ranking.

Sister-species errors (Pangshura tecta vs sylhetensis) and softshell
collapse (Lissemys vs Pelochelys) often jump across these groups. Scoring
the photograph against group text prompts first, then ranking only inside
the winning group(s), cuts those jumps.
"""

from __future__ import annotations

# species_id -> gate group
GROUP_OF: dict[str, str] = {}

_GROUPS: dict[str, list[str]] = {
    "softshell": [
        "lissemys_punctata",
        "nilssonia_gangetica",
        "nilssonia_hurum",
        "nilssonia_leithii",
        "nilssonia_nigricans",
        "chitra_indica",
        "amyda_ornata",
        "pelochelys_cantorii",
    ],
    "pangshura": [
        "pangshura_tecta",
        "pangshura_tentoria",
        "pangshura_smithii",
        "pangshura_sylhetensis",
    ],
    "river_large": [
        "batagur_kachuga",
        "batagur_dhongoka",
        "batagur_baska",
        "hardella_thurjii",
    ],
    "black_box": [
        "melanochelys_trijuga",
        "melanochelys_tricarinata",
        "cuora_amboinensis",
        "cuora_mouhotii",
        "cyclemys_gemeli",
        "vijayachelys_silvatica",
        "morenia_petersi",
    ],
    "spotted_pond": [
        "geoclemys_hamiltonii",
    ],
    "tortoise": [
        "geochelone_elegans",
        "indotestudo_elongata",
        "indotestudo_travancorica",
        "manouria_emys",
        "manouria_impressa",
    ],
    "exotic_slider": [
        "trachemys_scripta_elegans",
    ],
}

for _group, _ids in _GROUPS.items():
    for _sid in _ids:
        GROUP_OF[_sid] = _group

# Text prompts for gating (BioCLIP text encoder).
GROUP_PROMPTS: dict[str, list[str]] = {
    "softshell": [
        "a softshell turtle with a leathery flat carapace",
        "Trionychidae softshell turtle flapshell or Nilssonia",
        "Indian flapshell turtle Lissemys punctata",
    ],
    "pangshura": [
        "a roofed turtle with a tented keeled carapace Pangshura",
        "Indian roofed or tent turtle with vertebral spike",
        "Pangshura tecta tentoria smithii freshwater turtle",
    ],
    "river_large": [
        "a large river terrapin Batagur or Hardella crowned river turtle",
        "red-crowned or three-striped roofed turtle Batagur",
    ],
    "black_box": [
        "a dark freshwater turtle Melanochelys or Cuora box turtle",
        "Indian black turtle with three keels",
    ],
    "spotted_pond": [
        "spotted pond turtle Geoclemys hamiltonii black shell yellow spots",
        "Geoclemys hamiltonii yellow spotted head",
    ],
    "tortoise": [
        "a terrestrial tortoise with high domed shell Geochelone or Indotestudo",
        "Indian star tortoise Geochelone elegans",
    ],
    "exotic_slider": [
        "red-eared slider Trachemys scripta elegans with red ear stripe",
        "invasive pond slider turtle with yellow stripes on neck",
    ],
}


def species_in_groups(groups: list[str]) -> set[str]:
    allowed: set[str] = set()
    for group in groups:
        allowed.update(_GROUPS.get(group, []))
    return allowed
