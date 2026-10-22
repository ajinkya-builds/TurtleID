"""Diagnostic trait text prompts for soft BioCLIP / BioCAP scoring."""

from __future__ import annotations

# Extra morphological prompts beyond scientific / common name.
# Tuned to the failure modes on the Turtles/ seizure set.
TRAIT_PROMPTS: dict[str, list[str]] = {
    "geochelone_elegans": [
        "Indian star tortoise with yellow star rays on each carapace scute",
        "high domed terrestrial tortoise with radiating star pattern",
        "Geochelone elegans yellow black star shell",
    ],
    "geoclemys_hamiltonii": [
        "spotted pond turtle with yellow spots on black head and shell",
        "Geoclemys hamiltonii three keeled dark carapace white streaks",
        "black freshwater turtle with irregular yellow head spots",
    ],
    "hardella_thurjii": [
        "crowned river turtle Hardella with bold yellow head stripes",
        "large dark river turtle with striped head Hardella thurjii",
        "Hardella thurjii three keeled brown carapace",
    ],
    "lissemys_punctata": [
        "Indian flapshell turtle Lissemys punctata soft leathery shell",
        "softshell turtle with femoral flaps covering hind limbs",
        "Lissemys punctata olive softshell with yellow spots sometimes",
    ],
    "nilssonia_gangetica": [
        "Indian softshell turtle Nilssonia gangetica without femoral flaps",
        "Nilssonia gangetica elongated softshell with tubular snout",
        "Ganges softshell turtle leathery carapace no flapshell hinges",
    ],
    "pangshura_tecta": [
        "Indian roofed turtle Pangshura tecta with spined third vertebral",
        "Pangshura tecta strongly tented keeled carapace red crown mark",
        "roofed turtle with elevated vertebral keel spike",
    ],
    "pangshura_tentoria": [
        "Indian tent turtle Pangshura tentoria with lower vertebral keel",
        "Pangshura tentoria tented carapace pinkish marginal ring sometimes",
        "tent turtle hard shell keeled but less spined than tecta",
    ],
    "trachemys_scripta_elegans": [
        "red-eared slider Trachemys scripta elegans red stripe behind eye",
        "invasive pond slider with yellow neck stripes and red ear patch",
        "Trachemys scripta elegans freshwater turtle red postorbital mark",
    ],
    "melanochelys_trijuga": [
        "Indian black turtle Melanochelys trijuga dark shell three keels",
        "Melanochelys trijuga olive black freshwater turtle",
    ],
    "cuora_amboinensis": [
        "Southeast Asian box turtle Cuora amboinensis hinged plastron",
    ],
    "cuora_mouhotii": [
        "keeled box turtle Cuora mouhotii jagged carapace",
    ],
    "chitra_indica": [
        "narrow-headed softshell Chitra indica very flat leathery shell",
    ],
    "pelochelys_cantorii": [
        "Asian giant softshell Pelochelys cantorii broad head",
    ],
    "amyda_ornata": [
        "Asiatic softshell Amyda ornata",
    ],
    "batagur_kachuga": [
        "red-crowned roofed turtle Batagur kachuga",
    ],
    "batagur_baska": [
        "northern river terrapin Batagur baska",
    ],
}


def prompts_for(species_id: str, scientific_name: str, common_en: str, family: str) -> list[str]:
    base = [
        f"Animalia Chordata Reptilia Testudines {family} {scientific_name}",
        scientific_name,
        common_en,
    ]
    return base + TRAIT_PROMPTS.get(species_id, [])
