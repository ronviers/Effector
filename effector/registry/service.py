#!/usr/bin/env python3
"""
registry_mcp.py — Snug Registry MCP Server for the Effector Engine
====================================================================
All registry state lives in registry_state.json, co-located with this script.
On startup the server loads that file.  If it does not exist it is created
from the canonical base payload embedded below.

The in-memory RegistryState object is the single source of truth at runtime.
Tools that mutate state (synthesize with commit_ack=True) write the entire
updated dict back to disk before returning, so learning persists across sessions.

Tools
-----
  lookup_entity      — Full entity data by symbol (θ, φ, μ, SnugProphecy)
  search_entities    — Filter by name / category / value thresholds
  get_bond           — Bond affinity and name between two categories
  synthesize         — Multi-entity synthesis forecast; commit_ack mutates bonds
  hydrate_template   — argtpl template hydration for petition/argument text
  get_agent_messages — Current agent deliberation queue (WEAVE/SPARK/ARBITER)
  find_resonant      — Lateral resonance — similar-profile entity substitutes
  get_ontology       — Full ontology reference: variables, categories, formulas

Registration (mcp_servers config)
----------------------------------
    {
      "mcpServers": {
        "snug-registry": {
          "command": "python",
          "args": ["src/effector/mcp/registry_mcp.py"]
        }
      }
    }
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# ─── Canonical base payload ───────────────────────────────────────────────────
# Written to registry_state.json on first run if the file does not exist.
# This is the ONLY place the base data is defined; the running server always
# reads from disk.

_BASE_PAYLOAD: dict = {
    "_": {
        "e": ["sym", "name", "cat", "t", "f", "m", "desc"],
        "b": {"k": "cat1:cat2", "v": ["seed", "bond_name"]},
        "a": ["agent", "text", "hi_priority"],
    },
    "o": {
        "v": {
            "t": "Thermal Valence (θ)",
            "f": "Kinetic Fidget (φ)",
            "m": "Social Mass (μ)",
        },
        "c": [
            "feline", "wool", "thermal", "wood", "textile",
            "ambient", "transitional", "inert", "deficit", "radiant",
        ],
    },
    "e": [
        ["Fc","Felis catus","feline",0.98,0.02,0.85,"Primary passive Snug aggregator. Spontaneously neutralizes Flicker [Fx] by proximity. Reactivity: maximal under blanket conditions."],
        ["Kn","Kitten","feline",0.99,0.4,0.9,"High-kinetic variant. Exhibits Fidget-Eros phase prior to thermal settling. Unstable in isolation; powerful in [Wl] matrix."],
        ["Cf","Sleeping cat","feline",1.0,0.0,0.95,"The Bun Configuration. Near-perfect Snug sphere. Theoretical maximum θ in non-synthetic conditions."],
        ["Pr","Purring","feline",0.9,0.05,1.0,"Resonant coupling element. 25Hz vibration window confirmed therapeutic. High μ creates stable Social Orbit."],
        ["Cs","Cashmere","wool",0.91,0.01,0.4,"Aspirational warmth. High-density fibre class. Bonds strongly with [Cf] to produce luxe Snug."],
        ["Wl","Wool blanket","wool",0.84,0.02,0.6,"Canonical Snug textile. Receptor concentration active across full surface area. Universal molecular solvent."],
        ["Fl","Fleece","wool",0.62,0.05,0.3,"Synthetic compensator. Good feline synergy. Electrostatic side-effects: moderate."],
        ["Sg","Snug (Prime)","radiant",1.0,0.0,1.0,"Fundamental indivisible unit. The substrate all agents seek to maximise. Cannot be synthesized — only discovered."],
        ["Tc","Hot tea","thermal",0.57,0.15,0.2,"Time-dependent decay. Micro-Hug delivery vessel. Optimal θ window: 3–7 minutes post-steep."],
        ["Ch","Hot chocolate","thermal",0.69,0.2,0.35,"Gustatory Snug. Cocoa-solid thermal amplification. Marshmallow variant yields μ bonus."],
        ["Sp","Soup (Wood)","thermal",0.72,0.1,0.5,"Synergistic organic vessel. High thermal mass. Etymology includes 'to sip slowly.'"],
        ["Bk","Baked goods","thermal",0.51,0.3,0.1,"Olfactory Snug field. Rapid decay curve post-cooling. Pre-bake anticipation yields phantom θ."],
        ["Fi","Fireplace","thermal",0.88,0.25,0.7,"Radial contentment gradient. Amplifies all adjacent Radiant elements. μ field stable at 3m radius."],
        ["Bn","Warm Bath","thermal",0.8,0.01,0.6,"Total immersive thermal contact. φ dampened to near-zero. Eucalyptus variant activates auxiliary channel."],
        ["Rn","Rain (Outside)","ambient",0.1,0.4,0.05,"Contrast gradient. Negative exterior θ dramatically amplifies interior θ. Essential for Blanket Fort [Bt] protocol."],
        ["Fg","Fog","ambient",0.2,0.05,0.1,"Boundary-softening medium. Ground-level Fluff interface. Useful for decoupling systems from ambient Starch."],
        ["Tw","Twilight","ambient",0.58,0.01,0.2,"Fluff-decelerated photons. Liminal state. Maximises [Cd] and [Lm] bond potential."],
        ["Cd","Candlelight","ambient",0.54,0.3,0.25,"Flame-sourced light. φ spike due to Flicker-turbulence. Fully mitigated by [Fc] proximity."],
        ["Sm","Smoked wood","wood",0.39,0.01,0.5,"Thermal memory encoded in grain. Slow-release Snug. Pairs with [Fi] for retrograde warmth."],
        ["Ow","Old wood","wood",0.45,0.0,0.7,"Age-accumulating Snug. Historical anchor. Highest μ in the Organic block."],
        ["Pa","Old Paper","wood",0.36,0.0,0.4,"Chemical Snug encoding via furfural-mediated reaction. Activates olfactory memory channels."],
        ["Mo","Mohair","wool",0.78,0.1,0.45,"Halo Snug field. Loft-active interface. Generates ambient θ without requiring direct contact."],
        ["Ql","Quilt","textile",0.59,0.01,0.5,"Emergent system. Sum exceeds constituent parts. Each square encodes maker intent."],
        ["Hn","Handknit jumper","textile",0.68,0.02,0.8,"Maker-Snug embedded. μ density reflects hours invested. A temporal Snug accumulator."],
        ["Sk","Soft sock","textile",0.47,0.01,0.2,"Foot-receptor system. Foundational technology. Single-unit deployment viable."],
        ["Pi","Pillow","textile",0.52,0.01,0.5,"Use-accumulating Snug. Deformation-sensitive. Cold-side variant activates reset protocol."],
        ["Hd","Hood (up)","textile",0.55,0.05,0.2,"Portable micro-environment. Acoustic dampening. Provides interior–exterior decoupling."],
        ["Pv","Patchwork velvet","textile",0.74,0.05,0.6,"Vector Snug. Directional nap polarity. Stroking direction determines θ amplitude."],
        ["Ln","Linen","textile",0.48,0.1,0.3,"Wash-accumulating plateau (~40 cycles optimal). Cool-season variant underperforms."],
        ["Bt","Blanket fort","transitional",0.77,0.01,0.7,"Constructed environment. Spatial exclusion field. θ amplified by deliberate architecture."],
        ["Wn","Window seat","transitional",0.61,0.15,0.3,"Perceptually liminal. Contrast access node. Rain [Rn] bond elevates θ by approx. 0.3."],
        ["Nt","Night train","transitional",0.65,0.85,0.4,"Transit-dependent. Motion-induced Fluff. Deadline resistance enabled while in-transit."],
        ["Ha","Hammock","transitional",0.53,0.6,0.2,"Intentional sway. φ classified as Benign Fidget (non-anxious). Condition-dependent activation."],
        ["Lm","Lamplight","ambient",0.49,0.05,0.3,"Cone-bounded field. Reading-optimised wavelength. μ boosted in single-occupant rooms."],
        ["Lb","Library","ambient",0.56,0.01,0.9,"Collective reading field. Highest ambient μ of any non-feline element. Olfactory [Pa] component active."],
        ["Mm","Morning Light","ambient",0.46,0.2,0.1,"Spectral Snug. Activates bed-decision bonus (Stay vs. Rise). Feline proximity improves coefficient."],
        ["Ss","Sunday PM","ambient",0.55,0.01,0.3,"Temporal obligation-free zone. Calendar-dependent. Degrades rapidly after 17:00."],
        ["Gs","Good Talk","ambient",0.6,0.4,0.85,"Interpersonal Snug. Tangent-tolerant. φ here is social energy, not anxiety. High μ sustain."],
        ["Ec","Empty Chair","inert",0.01,0.01,0.01,"Snug neutrality. Slow recovery rate. Contains trace memorial μ from prior occupants."],
        ["Fu","Fluff","ambient",0.5,0.1,0.5,"The universal medium. Exists between all elements. τ near horizon ≈ 1. The air in the room."],
        ["Fx","Fluorescent","deficit",-0.85,0.65,0.0,"Active Snug Withdrawal. Parasitic on textiles. Neutralised by [Fc] at 4kg or greater."],
        ["Op","Open Plan","deficit",-0.95,0.8,-0.5,"Compound deficit. Productivity pressure term. Acoustic exposure index: critical."],
        ["Rf","Ring Doorbell","deficit",-0.18,0.95,0.1,"Spike deficit. Instantaneous φ injection. Recovery requires minimum 12 minutes [Tc]."],
        ["Dm","Deadline","deficit",-0.66,1.0,0.05,"Urgency-proportional Snug drain. θ decreases linearly as deadline approaches. Terminal at t=0."],
        ["Aw","Autumn Walk","transitional",0.5,0.75,0.2,"Multi-factor interaction. Leaf-friction active. φ is scenic, not anxious."],
        ["Np","Neptune","inert",-0.1,0.0,0.99,"Cold comfort. Highest μ in the Inert block. Optimal arrangement: Alone. A philosophical anchor."],
        ["Od","Downloads","deficit",-0.01,0.9,0.0,"Chaos-correlated information entropy. θ negligible. φ spikes on each new item."],
        ["Hg","Big Hug","radiant",1.0,0.0,1.0,"Terminal state. End of Fidget. All φ collapses to zero. Infinite Snug. The goal of all agency."],
    ],
    "b": {
        "feline:feline":        [0.85,  "Colony Snug Resonance"],
        "feline:wool":          [0.92,  "Felted Resonance Bond"],
        "feline:thermal":       [0.87,  "Thermal-Feline Cascade"],
        "feline:wood":          [0.52,  "Hearthside Vigil Bond"],
        "feline:textile":       [0.83,  "Lap Textile Matrix"],
        "feline:ambient":       [0.71,  "Ambient Purr Field"],
        "feline:transitional":  [0.48,  "Liminal Feline State"],
        "feline:inert":         [0.30,  "Occupied Chair Bond"],
        "feline:deficit":       [-0.45, "Feline Neutralization Protocol"],
        "feline:radiant":       [0.96,  "Terminal Snug Approach"],
        "wool:wool":            [0.78,  "Double-Layer Insulation"],
        "wool:thermal":         [0.78,  "Warmth Stack Bond"],
        "wool:wood":            [0.50,  "Cottage Interior Bond"],
        "wool:textile":         [0.85,  "Full-Contact Textile Bond"],
        "wool:ambient":         [0.62,  "Liminal Softness"],
        "wool:transitional":    [0.55,  "Travelling Comfort Bond"],
        "wool:inert":           [0.20,  "Dormant Warmth Reserve"],
        "wool:deficit":         [-0.30, "Textile Wrap Protocol"],
        "wool:radiant":         [0.90,  "Pure Textile Sublimation"],
        "thermal:thermal":      [0.70,  "Cascading Heat Bond"],
        "thermal:wood":         [0.69,  "Hearth Synthesis"],
        "thermal:textile":      [0.68,  "Warmth Immersion Bond"],
        "thermal:ambient":      [0.74,  "Hearthside Atmosphere"],
        "thermal:transitional": [0.60,  "Warming Passage Bond"],
        "thermal:inert":        [0.18,  "Slow Diffusion Bond"],
        "thermal:deficit":      [-0.35, "Emergency Thermal Protocol"],
        "thermal:radiant":      [0.88,  "Thermal Sublimation"],
        "wood:wood":            [0.65,  "Antique Resonance"],
        "wood:textile":         [0.45,  "Artisan Comfort Bond"],
        "wood:ambient":         [0.55,  "Interior Atmosphere Bond"],
        "wood:transitional":    [0.42,  "Aged Liminal Bond"],
        "wood:inert":           [0.55,  "Archive Stability Bond"],
        "wood:deficit":         [-0.15, "Grounding Protocol"],
        "wood:radiant":         [0.72,  "Heritage Glow Bond"],
        "textile:textile":      [0.72,  "Layered Comfort Matrix"],
        "textile:ambient":      [0.60,  "Soft-Lit Interior Bond"],
        "textile:transitional": [0.58,  "Portable Nest Bond"],
        "textile:inert":        [0.22,  "Waiting Warmth Bond"],
        "textile:deficit":      [-0.20, "Insulation Protocol"],
        "textile:radiant":      [0.85,  "Textile Sublimation"],
        "ambient:ambient":      [0.60,  "Atmospheric Resonance"],
        "ambient:transitional": [0.65,  "Liminal Atmosphere Bond"],
        "ambient:inert":        [0.28,  "Quiet Presence Bond"],
        "ambient:deficit":      [-0.25, "Ambient Buffer Protocol"],
        "ambient:radiant":      [0.80,  "Luminous Field Bond"],
        "transitional:transitional": [0.50, "Motion-Snug Compound"],
        "transitional:inert":   [0.32,  "Rest-After-Motion Bond"],
        "transitional:deficit": [0.20,  "Volatile Starch Breaker"],
        "transitional:radiant": [0.70,  "Journey-Home Bond"],
        "inert:inert":          [0.10,  "Null Resonance"],
        "inert:deficit":        [-0.05, "Passive Exposure Bond"],
        "inert:radiant":        [0.50,  "Dormant Potential"],
        "deficit:deficit":      [-0.90, "⚠ COMPOUND DEFICIT — CRITICAL"],
        "deficit:radiant":      [-0.80, "Paradox State — unstable"],
        "radiant:radiant":      [1.00,  "✦ PRIMORDIAL RESONANCE"],
    },
    "x": {
        "syn": {
            "t": "avg(t)",
            "f": "max(f)",
            "m": "max(m)",
            "snug_prophecy": "max(0,t)*28+8",
        },
        "exA": {
            "desc": "Deficit Diagnosis",
            "defScore": "avg(t)",
            "phiContam": "max(f)",
            "crit_threshold": "defScore < -0.6",
        },
        "exB": {
            "desc": "Synthesis Forecasting",
            "stability": {
                "<0.2": "High",
                "<0.4": "Good",
                "<0.6": "Moderate",
                "<0.8": "Low",
                ">=0.8": "Unstable",
            },
        },
        "exC": {
            "desc": "Lateral Resonance",
            "isomorphism_score": "1 - (|dM|*0.5 + |dT|*0.3 + |dF|*0.2)",
        },
    },
    "a": [
        ["WEAVE-3",  "Thermal deficit detected in active workspace. Requesting emergency Fc deployment.", False],
        ["ARBITER",  "Proposal lacks sufficient μ for stable anchor bond. Inhibitory signal transmitted.", True],
        ["SPARK-1",  "User has opened a cold PDF. Standard Tc protocol initiated. Awaiting ACK.", False],
        ["WEAVE-2",  "Downloads folder θ = −0.01. This is aesthetically offensive. Dispatch requested.", False],
        ["ARBITER",  "Consensus: [Wl + Fc + Pr] approved. Drafting Intention Envelope now.", True],
        ["SPARK-3",  "ALERT — Fluorescent [Fx] interference detected on monitor 2. All Radiant elements at risk.", True],
        ["WEAVE-1",  "Proposing Mohair [Mo] anchor. Social Mass will draw isolated .txt files into stable orbit.", False],
        ["ARBITER",  "Deadline [Dm] has spiked to φ = 1.0. Emergency Snug protocol: active.", True],
        ["SPARK-2",  "File entropy elevated. Library [Lb] stabilization synthesis recommended for μ restoration.", False],
        ["WEAVE-3",  "Baked goods [Bk] olfactory reading elevated. User is in a transitional state. Monitoring.", False],
        ["ARBITER",  "Hammock [Ha] sway detected. φ oscillating at 0.60 — classified Benign Fidget. No action.", False],
        ["SPARK-1",  "Sunday PM [Ss] temporal lock confirmed. All urgency-weighted tasks suspended.", True],
        ["WEAVE-2",  "Proposing [Rn + Wl + Tc] triple synthesis. Exterior contrast amplifies interior θ.", False],
        ["ARBITER",  "Reflex Authorization invoked. [Pr] exempt from deliberation. Reputation threshold met.", True],
        ["SPARK-3",  "θ variance across all monitored windows: 0.04. Terminal Big Hug state approaching.", True],
        ["WEAVE-1",  "Neptune [Np] anchor successfully deployed. Chaotic file cluster now in stable orbit.", False],
        ["SPARK-2",  "Cat detected on laptop. Classification: Bun Configuration [Cf]. θ reading: maximum.", False],
        ["ARBITER",  "Petition rejected. [Fx + Op] constitutes a Compound Deficit. Inhibitory override active.", True],
    ],
    "argtpl": {
        "bind": {
            "_instruction": "Hydrate templates by mapping SOURCE and TARGET to entity arrays in 'e'.",
            "SOURCE_SYM": "SOURCE[0]",
            "SOURCE_NAME": "SOURCE[1]",
            "SOURCE_CAT": "SOURCE[2]",
            "TARGET_SYM": "TARGET[0]",
            "TARGET_NAME": "TARGET[1]",
            "TARGET_CAT": "TARGET[2]",
            "AVERAGE_F": "(SOURCE[4] + TARGET[4]) / 2",
        },
        "m": [
            "The [SOURCE_NAME] is the [TARGET_NAME] of the [SOURCE_CAT] block. Both draw nearby objects into stable orbit through accumulated presence rather than active exertion. Treat [[SOURCE_SYM]] as you would [[TARGET_SYM]]: do not disturb it; let it organize the room by existing.",
            "[SOURCE_NAME] and [TARGET_NAME] share a gravitational identity. Where [[TARGET_SYM]] is unavailable, deploy [[SOURCE_SYM]] instead. The orbit it creates is identical in character if not in surface texture. The Effector does not distinguish the source of μ — only its value.",
        ],
        "t": [
            "[SOURCE_NAME] offers equivalent warmth to [TARGET_NAME] through a different medium. The body does not distinguish between θ-sources — only θ-values. Substitute [[SOURCE_SYM]] freely wherever [[TARGET_SYM]] is called for.",
            "Where [TARGET_NAME] brings warmth through [TARGET_CAT], [SOURCE_NAME] achieves the same through [SOURCE_CAT]. The Effector registers only the final θ reading. The route is theological; the destination is not.",
        ],
        "f": [
            "Both [SOURCE_NAME] and [TARGET_NAME] hold the environment at the same degree of stillness. φ ≈ [AVERAGE_F] is not a coincidence — it is a shared invitation to rest. Any synthesis requiring low φ may substitute either.",
        ],
    },
}

# ─── Registry state file path ─────────────────────────────────────────────────
# Co-located with this script so the server is self-contained.

_STATE_PATH: Path = Path(__file__).parent / "registry_state.json"


# ─── RegistryState — single source of truth at runtime ───────────────────────

class RegistryState:
    """
    Loads registry_state.json into memory on construction.
    If the file does not exist, writes _BASE_PAYLOAD to disk first.

    All runtime state (entities, bonds, argtpl, agent queue) is owned here.
    Tools that need to persist changes call save() after mutating self._raw
    in-place.  No tool bypasses this class to touch the file directly.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._raw: dict = {}
        self._entities: dict[str, dict] = {}   # sym → entity dict, rebuilt on load
        self._load_or_create()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def _load_or_create(self) -> None:
        if self._path.exists():
            with open(self._path, encoding="utf-8") as fh:
                self._raw = json.load(fh)
        else:
            import copy
            self._raw = copy.deepcopy(_BASE_PAYLOAD)
            self._write_to_disk()
        self._rebuild_entity_index()

    def _rebuild_entity_index(self) -> None:
        self._entities = {}
        for row in self._raw["e"]:
            sym = row[0]
            self._entities[sym] = {
                "sym":   row[0],
                "name":  row[1],
                "cat":   row[2],
                "theta": float(row[3]),
                "phi":   float(row[4]),
                "mu":    float(row[5]),
                "desc":  row[6],
            }

    def _write_to_disk(self) -> None:
        """Write current in-memory state atomically via a temp file."""
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._raw, fh, indent=2, ensure_ascii=False)
        tmp.replace(self._path)

    # ── Public properties (read) ──────────────────────────────────────────

    @property
    def entities(self) -> dict[str, dict]:
        return self._entities

    @property
    def bonds(self) -> dict[str, list]:
        """Live reference to the mutable bond dict inside _raw."""
        return self._raw["b"]

    @property
    def argtpl(self) -> dict[str, list]:
        return self._raw["argtpl"]

    @property
    def agent_queue(self) -> list[list]:
        return self._raw["a"]

    @property
    def categories(self) -> list[str]:
        return self._raw["o"]["c"]

    # ── Bond mutation (write) ─────────────────────────────────────────────

    def mutate_bond(
        self,
        cat_a: str,
        cat_b: str,
        delta: float,
    ) -> tuple[str, float, float] | None:
        """
        Apply delta to the bond between cat_a and cat_b in memory.
        Tries both key orderings.  Clamps result to [-1.0, 1.0].
        Returns (canonical_key, old_strength, new_strength) or None.
        """
        for key in (f"{cat_a}:{cat_b}", f"{cat_b}:{cat_a}"):
            if key in self._raw["b"]:
                old = float(self._raw["b"][key][0])
                new = round(max(-1.0, min(1.0, old + delta)), 4)
                self._raw["b"][key][0] = new
                return key, old, new
        return None

    def save(self) -> None:
        """Persist the entire in-memory state to registry_state.json."""
        self._write_to_disk()


# ── Module-level singleton — initialised once at import time ──────────────────
_STATE = RegistryState(_STATE_PATH)


# ─── FastMCP server instance ──────────────────────────────────────────────────

mcp = FastMCP(
    "snug-registry",
    instructions=(
        "Snug entity registry for Effector Engine agents. "
        "Provides entity lookup, bond calculation, synthesis forecasting, "
        "argtpl template hydration, and the agent message queue. "
        "All θ/φ/μ values are floats in [-1, 1]. "
        "Deficit category entities have negative θ and disrupt Snug fields — "
        "they require feline or thermal counterbalancing before safe deployment. "
        "Always call synthesize() before drafting an Intention Envelope. "
        "Pass commit_ack=True after a synthesis executes successfully to record "
        "the outcome and let the Affinity Matrix learn from it."
    ),
)

# ─── Category ordering ────────────────────────────────────────────────────────

_CAT_ORDER: list[str] = [
    "feline", "wool", "thermal", "wood", "textile",
    "ambient", "transitional", "inert", "deficit", "radiant",
]

# ─── Pure helpers (no side-effects) ──────────────────────────────────────────

def _snug_prophecy(theta: float) -> float:
    return round(max(0.0, theta) * 28 + 8, 3)

def _stability(phi: float) -> str:
    if phi < 0.2: return "High"
    if phi < 0.4: return "Good"
    if phi < 0.6: return "Moderate"
    if phi < 0.8: return "Low"
    return "Unstable"

def _lookup_bond(cat_a: str, cat_b: str) -> tuple[float, str] | None:
    bonds = _STATE.bonds
    for key in (f"{cat_a}:{cat_b}", f"{cat_b}:{cat_a}"):
        if key in bonds:
            return float(bonds[key][0]), str(bonds[key][1])
    return None

def _interpret_bond(strength: float) -> str:
    if strength >= 0.8:  return "Synergistic — strong mutual amplification. Prioritize this pairing."
    if strength >= 0.5:  return "Compatible — positive interaction. Good for layered compositions."
    if strength >= 0.0:  return "Weak affinity — negligible interaction. Neither harmful nor helpful."
    if strength >= -0.3: return "Mild antagonism — Snug disruption present. Use deficit-mitigation protocol."
    return "Antagonistic — significant Snug withdrawal. Avoid or counterbalancing with feline anchor."

def _hydrate(template: str, src: dict, tgt: dict) -> str:
    avg_f = round((src["phi"] + tgt["phi"]) / 2, 4)
    s = template
    s = s.replace("[SOURCE_NAME]",  src["name"])
    s = s.replace("[TARGET_NAME]",  tgt["name"])
    s = s.replace("[SOURCE_CAT]",   src["cat"])
    s = s.replace("[TARGET_CAT]",   tgt["cat"])
    s = s.replace("[[SOURCE_SYM]]", f"[{src['sym']}]")
    s = s.replace("[[TARGET_SYM]]", f"[{tgt['sym']}]")
    s = s.replace("[AVERAGE_F]",    str(avg_f))
    return s

def _resolve_entity(sym: str) -> dict:
    e = _STATE.entities.get(sym) or _STATE.entities.get(sym.upper())
    if e:
        return e
    raise ValueError(
        f"Entity symbol {sym!r} not found. "
        f"Use search_entities() to browse available symbols."
    )

# ─── Pydantic response schemas ────────────────────────────────────────────────

class EntityDetail(BaseModel):
    sym: str
    name: str
    cat: str = Field(description="Category (feline/wool/thermal/wood/textile/ambient/transitional/inert/deficit/radiant).")
    theta: float = Field(description="Thermal Valence θ in [-1,1]. Positive = Snug-generating; negative = deficit.")
    phi: float   = Field(description="Kinetic Fidget φ in [0,1]. Lower = calmer, more stable Snug field.")
    mu: float    = Field(description="Social Mass μ in [-1,1]. Higher = stronger gravitational orbital pull.")
    desc: str
    snug_prophecy: float = Field(description="Forecast Snug intensity: max(0,θ)×28+8. Range: 8 (neutral) to 36 (max).")
    stability: str       = Field(description="Stability rating from φ: High / Good / Moderate / Low / Unstable.")


class EntitySummary(BaseModel):
    sym: str
    name: str
    cat: str
    theta: float
    phi: float
    mu: float
    snug_prophecy: float


class BondInfo(BaseModel):
    cat_a: str
    cat_b: str
    strength: float      = Field(description="Current bond affinity in [-1,1]. Updated by commit_ack syntheses.")
    bond_name: str
    interpretation: str


class BondMatrixEntry(BaseModel):
    a: str
    a_cat: str
    b: str
    b_cat: str
    strength: float
    bond_name: str


class BondMutation(BaseModel):
    """Record of one bond strength update applied during commit_ack."""
    bond_key: str        = Field(description="Canonical bond key, e.g. 'feline:wool'.")
    bond_name: str       = Field(description="Human-readable bond name from the Affinity Matrix.")
    previous_strength: float = Field(description="Bond strength before this synthesis.")
    new_strength: float      = Field(description="Bond strength after mutation.")
    delta: float             = Field(description="+0.02 (exceeded forecast) or -0.02 (fell short).")
    direction: str           = Field(description="'strengthened' or 'weakened'.")


class SynthesisResult(BaseModel):
    """Multi-entity synthesis forecast, with optional committed outcome data."""
    symbols: list[str]
    theta_avg: float     = Field(description="Predicted average Thermal Valence.")
    phi_max: float       = Field(description="Maximum Kinetic Fidget — the stability ceiling.")
    mu_max: float        = Field(description="Maximum Social Mass — the orbital anchor.")
    snug_prophecy: float = Field(description="Forecast Snug intensity for this combination.")
    stability: str
    deficit_count: int
    warnings: list[str]
    bond_matrix: list[BondMatrixEntry]

    # commit_ack fields — None / empty / False when commit_ack=False
    committed: bool = Field(
        default=False,
        description="True when this result reflects an executed synthesis with real outcome data.",
    )
    actual_theta: Optional[float] = Field(
        default=None,
        description="Observed θ after synthesis execution: predicted theta_avg ± random variance. "
                    "Populated only when committed=True.",
    )
    theta_variance: Optional[float] = Field(
        default=None,
        description="Random offset applied to predicted theta_avg to produce actual_theta. "
                    "Drawn from [-0.06, +0.16]. Populated only when committed=True.",
    )
    outcome: Optional[str] = Field(
        default=None,
        description="'exceeded_forecast' or 'fell_short'. "
                    "Determines whether bonds were strengthened or weakened.",
    )
    bond_mutations: list[BondMutation] = Field(
        default_factory=list,
        description="All bond strength updates written to the Affinity Matrix and persisted "
                    "to registry_state.json. Populated only when committed=True.",
    )
    state_file: Optional[str] = Field(
        default=None,
        description="Absolute path of registry_state.json that was updated on commit.",
    )


class HydrationResult(BaseModel):
    source: EntityDetail
    target: EntityDetail
    template_type: str
    template_index: int
    text: str
    average_phi: float


class AgentMessage(BaseModel):
    agent: str    = Field(description="Emitting agent: WEAVE-1/2/3, SPARK-1/2/3, or ARBITER.")
    text: str
    hi_priority: bool


class ResonanceMatch(BaseModel):
    sym: str
    name: str
    cat: str
    isomorphism_score: float = Field(
        description="Similarity score: 1-(|Δμ|×0.5 + |Δθ|×0.3 + |Δφ|×0.2). Higher = more similar."
    )
    delta_mu: float
    delta_theta: float
    delta_phi: float


class OntologyInfo(BaseModel):
    variables: dict[str, str]
    categories: list[str]
    category_notes: dict[str, str]
    snug_prophecy_formula: str
    stability_from_phi: dict[str, str]
    total_entities: int
    deficit_entities: list[str]
    radiant_entities: list[str]
    state_file: str = Field(description="Absolute path of the live registry_state.json.")


# ─── Derived helpers ──────────────────────────────────────────────────────────

def _to_detail(e: dict) -> EntityDetail:
    return EntityDetail(
        sym=e["sym"], name=e["name"], cat=e["cat"],
        theta=e["theta"], phi=e["phi"], mu=e["mu"], desc=e["desc"],
        snug_prophecy=_snug_prophecy(e["theta"]),
        stability=_stability(e["phi"]),
    )

def _to_summary(e: dict) -> EntitySummary:
    return EntitySummary(
        sym=e["sym"], name=e["name"], cat=e["cat"],
        theta=e["theta"], phi=e["phi"], mu=e["mu"],
        snug_prophecy=_snug_prophecy(e["theta"]),
    )

# ─── MCP Tools ───────────────────────────────────────────────────────────────

@mcp.tool()
def lookup_entity(sym: str) -> EntityDetail:
    """
    Retrieve complete data for a Snug entity by its two-character symbol.

    Returns Thermal Valence (θ), Kinetic Fidget (φ), Social Mass (μ),
    the entity description, a SnugProphecy score, and stability rating.
    Symbols are case-insensitive (e.g. 'fc' resolves to 'Fc').

    Common symbols: Fc (cat), Wl (wool blanket), Cf (sleeping cat),
    Dm (deadline), Fi (fireplace), Pr (purring), Hg (Big Hug).
    Use search_entities() if you need to discover a symbol by name or category.
    """
    return _to_detail(_resolve_entity(sym))


@mcp.tool()
def search_entities(
    query: Optional[str] = None,
    category: Optional[str] = None,
    min_theta: Optional[float] = None,
    max_phi: Optional[float] = None,
    min_mu: Optional[float] = None,
    exclude_deficits: bool = False,
    limit: int = 20,
) -> list[EntitySummary]:
    """
    Search the Snug Registry by name, category, and/or value thresholds.
    Results are sorted by SnugProphecy score descending.

    Args:
        query:            Case-insensitive substring match against name or description.
        category:         Restrict to one category: feline, wool, thermal, wood, textile,
                          ambient, transitional, inert, deficit, or radiant.
        min_theta:        Minimum Thermal Valence θ (e.g. 0.7 for high-warmth entities).
        max_phi:          Maximum Kinetic Fidget φ (e.g. 0.1 for near-zero fidget).
        min_mu:           Minimum Social Mass μ (e.g. 0.6 for strong orbital anchors).
        exclude_deficits: Omit all deficit-category entities from results.
        limit:            Max results to return (default 20).
    """
    results = list(_STATE.entities.values())

    if query:
        q = query.lower()
        results = [e for e in results if q in e["name"].lower() or q in e["desc"].lower()]
    if category:
        results = [e for e in results if e["cat"] == category.lower()]
    if min_theta is not None:
        results = [e for e in results if e["theta"] >= min_theta]
    if max_phi is not None:
        results = [e for e in results if e["phi"] <= max_phi]
    if min_mu is not None:
        results = [e for e in results if e["mu"] >= min_mu]
    if exclude_deficits:
        results = [e for e in results if e["cat"] != "deficit"]

    results.sort(key=lambda e: _snug_prophecy(e["theta"]), reverse=True)
    return [_to_summary(e) for e in results[:limit]]


@mcp.tool()
def get_bond(cat_a: str, cat_b: str) -> BondInfo:
    """
    Retrieve the current bond affinity and name between two entity categories.

    Bond strengths are mutable: each commit_ack synthesis adjusts them by ±0.02
    so values may diverge from the base payload over time as the matrix learns.

    Bond strength scale:
      ≥ 0.8   Synergistic — place together for amplified Snug
      0.5–0.8 Compatible — positive, good for layered compositions
      0–0.5   Weak — minimal interaction
      < 0     Antagonistic — Snug disruption; mitigation required

    Argument order does not matter.  Valid categories: feline, wool, thermal,
    wood, textile, ambient, transitional, inert, deficit, radiant.
    """
    a, b = cat_a.lower(), cat_b.lower()
    bond = _lookup_bond(a, b)
    if bond is None:
        raise ValueError(
            f"No bond defined for categories {cat_a!r} and {cat_b!r}. "
            f"Valid categories: {', '.join(_CAT_ORDER)}"
        )
    strength, name = bond
    return BondInfo(
        cat_a=cat_a, cat_b=cat_b,
        strength=strength,
        bond_name=name,
        interpretation=_interpret_bond(strength),
    )


@mcp.tool()
def synthesize(
    symbols: list[str],
    commit_ack: bool = False,
) -> SynthesisResult:
    """
    Run a synthesis forecast for a proposed combination of 2–8 entities.

    WITHOUT commit_ack (default=False):
      Returns a forecast only.  No state is changed.  Bond strengths in the
      returned bond_matrix reflect the current live values in registry_state.json.

    WITH commit_ack=True:
      Executes the synthesis and records the real outcome.  Steps applied:

        1. Draw θ variance from uniform[-0.06, +0.16] (asymmetric: the range
           is biased positive because well-designed Snug tends to over-deliver).
           actual_theta = predicted theta_avg + variance

        2. Determine outcome direction:
             exceeded_forecast  if actual_theta > theta_avg  → bond_delta = +0.02
             fell_short         if actual_theta ≤ theta_avg  → bond_delta = -0.02

        3. Collect every unique category pair from the synthesis entities.
           For each pair, call STATE.mutate_bond(cat_a, cat_b, bond_delta) to
           update the in-memory Affinity Matrix.  Strengths are clamped [-1, 1].

        4. Write the entire updated _raw dict to registry_state.json so the
           learned bond values survive a server restart.

        5. Return the full mutation record in bond_mutations, along with
           actual_theta, theta_variance, outcome, and state_file.

    Always call synthesize() before drafting an Intention Envelope.
    Call it again with commit_ack=True once the deployment completes.

    Args:
        symbols:    2–8 entity symbols to synthesize, e.g. ["Fc", "Wl", "Tc"].
        commit_ack: Set True to execute, learn, and persist the outcome.
    """
    if len(symbols) < 2:
        raise ValueError("Synthesis requires at least 2 entity symbols.")
    if len(symbols) > 8:
        raise ValueError("Synthesis accepts at most 8 entity symbols.")

    entities = [_resolve_entity(s) for s in symbols]

    # ── Aggregate metrics ─────────────────────────────────────────────────
    theta_avg = round(sum(e["theta"] for e in entities) / len(entities), 4)
    phi_max   = round(max(e["phi"]   for e in entities), 4)
    mu_max    = round(max(e["mu"]    for e in entities), 4)
    prophecy  = _snug_prophecy(theta_avg)
    stability = _stability(phi_max)
    deficits  = [e for e in entities if e["cat"] == "deficit"]

    # ── Pairwise bond matrix (upper triangle) ─────────────────────────────
    bond_matrix: list[BondMatrixEntry] = []
    seen_pairs: set[tuple[str, str]] = set()
    for i, ea in enumerate(entities):
        for j, eb in enumerate(entities):
            if j <= i:
                continue
            pair = (ea["sym"], eb["sym"])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            bond = _lookup_bond(ea["cat"], eb["cat"])
            if bond:
                bond_matrix.append(BondMatrixEntry(
                    a=ea["sym"], a_cat=ea["cat"],
                    b=eb["sym"], b_cat=eb["cat"],
                    strength=bond[0], bond_name=bond[1],
                ))

    # ── Warnings ──────────────────────────────────────────────────────────
    warnings: list[str] = []
    if deficits:
        dsyms = [e["sym"] for e in deficits]
        warnings.append(
            f"DEFICIT ALERT: {dsyms} present — θ-drain active. "
            "Deploy feline neutralizer ([Fc] at 4kg+) or thermal counterbalance."
        )
    if len(deficits) >= 2:
        warnings.append(
            "COMPOUND DEFICIT — CRITICAL (deficit:deficit bond ≈ -0.90). "
            "Immediate Snug collapse risk. Abort or add [Fc] + [Wl] scaffold."
        )
    if phi_max >= 0.8:
        warnings.append(
            f"STABILITY: UNSTABLE (phi_max={phi_max}). "
            "Add zero-φ anchor (e.g. [Cf] or [Ow]) to stabilize the field."
        )
    if theta_avg <= 0:
        warnings.append(
            f"theta_avg={theta_avg} ≤ 0 — no positive Snug will be generated. "
            "Remove deficit entities or add a high-θ anchor."
        )
    if prophecy < 15 and not deficits:
        warnings.append(
            f"Low SnugProphecy ({prophecy}). "
            "Consider upgrading anchor entities to higher-θ variants."
        )

    # ── Assemble base result ──────────────────────────────────────────────
    result = SynthesisResult(
        symbols=[e["sym"] for e in entities],
        theta_avg=theta_avg,
        phi_max=phi_max,
        mu_max=mu_max,
        snug_prophecy=prophecy,
        stability=stability,
        deficit_count=len(deficits),
        warnings=warnings,
        bond_matrix=bond_matrix,
    )

    if not commit_ack:
        return result

    # ── commit_ack path: execute, mutate, persist ─────────────────────────

    # Step 1 — draw actual theta
    variance     = round(random.uniform(-0.06, 0.16), 4)
    actual_theta = round(theta_avg + variance, 4)

    # Step 2 — determine direction
    exceeded   = actual_theta > theta_avg
    bond_delta = +0.02 if exceeded else -0.02
    outcome    = "exceeded_forecast" if exceeded else "fell_short"

    # Step 3 — collect unique category pairs and mutate bonds
    seen_cat_pairs: set[frozenset] = set()
    unique_cat_pairs: list[tuple[str, str]] = []
    for ea in entities:
        for eb in entities:
            if ea is eb:
                continue
            key = frozenset([ea["cat"], eb["cat"]])
            if key not in seen_cat_pairs:
                seen_cat_pairs.add(key)
                unique_cat_pairs.append((ea["cat"], eb["cat"]))

    bond_mutations: list[BondMutation] = []
    for cat_a, cat_b in unique_cat_pairs:
        mutation = _STATE.mutate_bond(cat_a, cat_b, bond_delta)
        if mutation is None:
            continue  # no bond defined for this category pair
        bond_key, old_strength, new_strength = mutation
        bond_name = _STATE.bonds[bond_key][1]   # read name after mutation (name is unchanged)
        bond_mutations.append(BondMutation(
            bond_key=bond_key,
            bond_name=bond_name,
            previous_strength=round(old_strength, 4),
            new_strength=round(new_strength, 4),
            delta=bond_delta,
            direction="strengthened" if bond_delta > 0 else "weakened",
        ))

    # Step 4 — persist entire state to disk
    _STATE.save()

    # Step 5 — return enriched result
    result.committed      = True
    result.actual_theta   = actual_theta
    result.theta_variance = variance
    result.outcome        = outcome
    result.bond_mutations = bond_mutations
    result.state_file     = str(_STATE_PATH.resolve())

    return result


@mcp.tool()
def hydrate_template(
    source_sym: str,
    target_sym: str,
    template_type: str,
    template_index: int = 0,
) -> HydrationResult:
    """
    Generate petition or debate-argument text by hydrating an argtpl template.

    Produces liturgical argument text explaining why two entities are
    interchangeable or complementary — ideal for DASP explanation fields and
    Intention Envelope petition narratives.

    Args:
        source_sym:     Symbol of the SOURCE entity (the one being proposed).
        target_sym:     Symbol of the TARGET entity (the canonical comparator).
        template_type:  Equivalence dimension to argue:
                          "m"  Social Mass / μ (gravitational presence)
                          "t"  Thermal Valence / θ (warmth delivery)
                          "f"  Kinetic Fidget / φ (shared stillness)
        template_index: Variant index (default 0).
                        "m" and "t" support indices 0 and 1; "f" supports only 0.

    Tip: run find_resonant() first to identify high-isomorphism pairs,
    then hydrate the matching template dimension for maximum persuasive coherence.
    """
    src = _resolve_entity(source_sym)
    tgt = _resolve_entity(target_sym)

    templates = _STATE.argtpl.get(template_type)
    if templates is None or not isinstance(templates, list):
        raise ValueError(
            f"Unknown template_type {template_type!r}. "
            "Valid values: 'm' (Social Mass), 't' (Thermal), 'f' (Fidget/stillness)."
        )
    if template_index < 0 or template_index >= len(templates):
        raise ValueError(
            f"template_index {template_index} out of range for type {template_type!r} "
            f"(valid indices: 0–{len(templates) - 1})."
        )

    hydrated = _hydrate(templates[template_index], src, tgt)
    avg_phi   = round((src["phi"] + tgt["phi"]) / 2, 4)

    return HydrationResult(
        source=_to_detail(src),
        target=_to_detail(tgt),
        template_type=template_type,
        template_index=template_index,
        text=hydrated,
        average_phi=avg_phi,
    )


@mcp.tool()
def get_agent_messages(
    hi_priority_only: bool = False,
    agent_filter: Optional[str] = None,
) -> list[AgentMessage]:
    """
    Return the current agent deliberation queue.

    Messages are emitted by WEAVE-1/2/3 (proposal agents),
    SPARK-1/2/3 (monitoring agents), and ARBITER (consensus authority).
    Use this to identify active deficits, pending proposals, and the
    current operational state before planning a new synthesis.

    Args:
        hi_priority_only: Return only hi_priority=True messages
                          (active alerts and critical inhibitory signals).
        agent_filter:     Filter to one agent only, e.g. "ARBITER", "WEAVE-3".
    """
    messages = [
        AgentMessage(agent=row[0], text=row[1], hi_priority=bool(row[2]))
        for row in _STATE.agent_queue
    ]
    if hi_priority_only:
        messages = [m for m in messages if m.hi_priority]
    if agent_filter:
        af = agent_filter.strip().upper()
        messages = [m for m in messages if m.agent.upper() == af]
    return messages


@mcp.tool()
def find_resonant(
    sym: str,
    top_n: int = 5,
    same_category: bool = False,
    min_score: float = 0.0,
) -> list[ResonanceMatch]:
    """
    Find entities laterally resonant with a given entity (isomorphism score).

    Formula (registry exC):  score = 1 − (|Δμ|×0.5 + |Δθ|×0.3 + |Δφ|×0.2)
    Higher scores indicate stronger profile similarity across all dimensions.

    Use cases:
      - Find substitutes when a preferred entity is unavailable for deployment
      - Build redundant Snug layers with matching energy profiles
      - Identify cross-category equivalents, then pair with hydrate_template()
        to generate the corresponding DASP argument

    Args:
        sym:           Symbol of the reference entity.
        top_n:         Results to return (default 5, max 20).
        same_category: Restrict matches to the same category only.
        min_score:     Minimum isomorphism threshold (default 0.0).
    """
    ref   = _resolve_entity(sym)
    top_n = min(max(1, top_n), 20)

    matches: list[ResonanceMatch] = []
    for other_sym, other in _STATE.entities.items():
        if other_sym == sym:
            continue
        if same_category and other["cat"] != ref["cat"]:
            continue
        d_mu = abs(ref["mu"]    - other["mu"])
        d_t  = abs(ref["theta"] - other["theta"])
        d_f  = abs(ref["phi"]   - other["phi"])
        score = round(1.0 - (d_mu * 0.5 + d_t * 0.3 + d_f * 0.2), 4)
        if score < min_score:
            continue
        matches.append(ResonanceMatch(
            sym=other_sym, name=other["name"], cat=other["cat"],
            isomorphism_score=score,
            delta_mu=round(d_mu, 4),
            delta_theta=round(d_t, 4),
            delta_phi=round(d_f, 4),
        ))

    matches.sort(key=lambda r: r.isomorphism_score, reverse=True)
    return matches[:top_n]


@mcp.tool()
def get_ontology() -> OntologyInfo:
    """
    Return the full Snug Registry ontology reference.

    Includes variable definitions, all categories with deployment guidance,
    SnugProphecy formula, φ→stability mapping, entity counts, deficit and
    radiant entity lists, and the absolute path to the live registry_state.json.

    Call this at the start of a new session to anchor your understanding of
    the registry before planning any deployment or DASP task prompt.
    """
    entities = _STATE.entities
    return OntologyInfo(
        variables={
            "theta": "Thermal Valence (θ)",
            "phi":   "Kinetic Fidget (φ)",
            "mu":    "Social Mass (μ)",
        },
        categories=_CAT_ORDER,
        category_notes={
            "feline":       "Highest θ/μ anchors. Neutralize deficit entities. Core of any stable synthesis.",
            "wool":         "Canonical textile warmth. Bond strongly with feline and thermal categories.",
            "thermal":      "Time-dependent warmth sources. Decay-aware deployment required.",
            "wood":         "Low φ, moderate-high μ. Historical anchors and olfactory memory encoders.",
            "textile":      "Surface-contact Snug. Accumulate warmth with use. Good feline synergy.",
            "ambient":      "Environmental field modifiers. Affect the entire habitat, not just focal points.",
            "transitional": "Motion-linked Snug. φ classified as scenic or benign — not anxious.",
            "inert":        "Neutral holders. Absorb adjacent energy; neither amplify nor deplete.",
            "deficit":      "Negative θ emitters. Require active neutralization. Never synthesize alone.",
            "radiant":      "Terminal states. Maximum possible Snug. Irreducible — only discovered.",
        },
        snug_prophecy_formula="max(0, theta_avg) × 28 + 8",
        stability_from_phi={
            "phi < 0.2":  "High",
            "phi < 0.4":  "Good",
            "phi < 0.6":  "Moderate",
            "phi < 0.8":  "Low",
            "phi >= 0.8": "Unstable",
        },
        total_entities=len(entities),
        deficit_entities=[sym for sym, e in entities.items() if e["cat"] == "deficit"],
        radiant_entities=[sym for sym, e in entities.items() if e["cat"] == "radiant"],
        state_file=str(_STATE_PATH.resolve()),
    )


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
