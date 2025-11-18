from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import yaml
from rapidfuzz import process, fuzz

DEFAULT_ONTOLOGY_PATH = Path(os.environ.get("SMART_WALKER_CONFIG_DIR", Path(
    __file__).resolve().parents[2] / "config")) / "ontology.yaml"


@dataclass
class Mapped:
    canonical_class: str | None
    ontology_class: str


class OntologyMapper:
    def __init__(self, ontology_path: Path | None = None, sim_threshold: float = 80.0):
        self.path = ontology_path or DEFAULT_ONTOLOGY_PATH
        if not self.path.exists():
            raise FileNotFoundError(f"ontology.yaml not found at {self.path}")
        with self.path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.sim_threshold = sim_threshold

        # Build lookup tables
        self.ontology_buckets = {}       # canonical -> ontology_name
        self.prompts = []                # list of prompt strings
        self.prompt2canon = {}           # prompt string -> canonical
        self.synonyms = {k.lower(): v.lower()
                         for k, v in cfg.get("synonyms_to_canonical", {}).items()}

        for bucket in cfg["ontology"]:
            ont_name = bucket["name"]
            for c in bucket.get("canonical", []):
                c_low = c.lower()
                self.ontology_buckets[c_low] = ont_name
                # canonical itself should be searchable
                self.prompts.append(c_low)
                self.prompt2canon[c_low] = c_low
            for p in bucket.get("prompts", []):
                p_low = p.lower()
                self.prompts.append(p_low)
                # map prompt back to the FIRST canonical for this bucket
                if bucket.get("canonical"):
                    self.prompt2canon[p_low] = bucket["canonical"][0].lower()

        # Fast set for hazards (optional convenience)
        self.hazard_set = set([c.lower() for b in cfg["ontology"]
                              if b["name"] == "hazard" for c in b.get("canonical", [])])

    def map_label(self, raw_label: str) -> Mapped:
        """Return canonical_class + ontology_class for a raw detector label."""
        if not raw_label:
            return Mapped(None, "unknown_obstacle")
        s = raw_label.strip().lower()

        # 1) direct synonym map
        if s in self.synonyms:
            canon = self.synonyms[s]
            ont = self.ontology_buckets.get(canon, "unknown_obstacle")
            return Mapped(canon, ont)

        # 2) direct canonical
        if s in self.ontology_buckets:
            return Mapped(s, self.ontology_buckets[s])

        # 3) fuzzy match against prompts/canonicals
        # returns (match, score, idx)
        match = process.extractOne(s, self.prompts, scorer=fuzz.WRatio)
        if match and match[1] >= self.sim_threshold:
            canon = self.prompt2canon.get(match[0], None)
            if canon:
                ont = self.ontology_buckets.get(canon, "unknown_obstacle")
                return Mapped(canon, ont)

        # 4) fallback
        return Mapped(None, "unknown_obstacle")
