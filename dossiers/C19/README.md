# Einstein Galaxy Dossiers

**Project:** Cosmic Intelligence Lab  
**Maintainer:** Glenn Lindsey  
**AI Research Assistant:** Einstein

## Specification

Version: 0.1

Status: Active Development

Last Updated: 2026-08-01

## Long-Term Vision

Each dossier aims to become the definitive scientific record for a single astronomical object.

A completed dossier should allow a researcher to understand the object without repeatedly searching the literature. Every observational measurement, interpretation, image, catalogue entry, and published analysis should be traceable to its original source.

## Purpose

The **Einstein Galaxy Dossier** system is a structured, living knowledge base for individual astronomical objects studied within the **Cosmic Intelligence Lab**.

Unlike a traditional research notebook, each dossier is **object-centered** rather than project-centered. Every dossier gathers all available information about a single galaxy from the astronomical literature, public archives, survey catalogs, and subsequent analysis into one continuously maintained document.

The goal is to transform scattered astronomical observations into a coherent, evidence-based, traceable scientific record.

## Scope

Each dossier is centered on a single astronomical object.

The dossier integrates observational data, published literature,
derived physical properties, archival imagery, cross-identifications,
and AI-assisted analyses into one continuously evolving scientific record.

The dossier does not replace the original literature; it provides a
traceable synthesis of that literature.

---

## Guiding Principle

Every statement in a dossier must be independently verifiable.

Whenever possible, measurements should cite the original publication,
catalogue, or archive from which they were derived.

Interpretations should be clearly distinguished from observations.

Einstein may assist with synthesis, but the scientific provenance of
every factual statement must remain explicit.

---

# Philosophy

Each dossier is built around three principles:

1. **Evidence First**
   - Every measurement should be traceable to its original source.
   - Observations are separated from interpretation.

2. **Living Documents**
   - A dossier is never considered "finished."
   - New publications, observations, and analyses are incorporated over time.

3. **Human + AI Collaboration**
   - The dossiers are designed to be maintained jointly by the researcher and the AI research assistant **Einstein**.
   - Einstein assists with literature review, data extraction, cross-identification, and scientific synthesis while preserving complete provenance.

---

# Why Object-Centered?

Astronomical knowledge is traditionally organized around publications.

The Einstein Galaxy Dossier system instead organizes knowledge around
the astronomical object itself.

Each dossier becomes the cumulative scientific history of one object,
bringing together observations from multiple facilities, surveys, and
decades of research into a single continuously evolving record.

---

# Scientific Writing Policy

Einstein may assist in:

- summarizing literature
- extracting measurements
- generating tables
- identifying cross-references
- suggesting future observations

Einstein does not replace peer-reviewed literature.

Every scientific statement incorporated into a dossier must remain
traceable to an identifiable source.

---

# Future Components

Future dossier directories may include:

C19.ipynb
    Research notebook.

C19.bib
    Bibliography.

C19.json
    Exchange format.

C19.fits
    Derived observational products.

C19.pdf
    Published dossier export.

---

# Directory Structure

Each astronomical object receives its own directory.

Example:

dossiers/

    C19/

        C19.md

        C19.yaml

        figures/

        images/

        notes/

        papers/

---

# Standard Contents

## C19.md

The primary human-readable scientific dossier.

Contains:

- Executive Summary
- Identification
- Discovery Timeline
- Observational Data
- Derived Physical Properties
- Imaging
- Spectroscopy
- Literature Evidence
- Scientific Interpretation
- Open Questions
- Provenance
- Einstein Research Notes

---

## C19.yaml

Machine-readable metadata used by Einstein.

Contains structured quantities such as

- identifiers
- coordinates
- redshift
- stellar mass
- star formation rate
- molecular gas masses
- references

This file is intended for future automation.

---

## figures/

Figures extracted from publications or generated during analysis.

Examples:

- ALMA continuum maps
- HST images
- MUSE overlays
- plots created during analysis

---

## images/

General imagery associated with the object.

May include

- finder charts
- archive images
- annotated images
- multiwavelength composites

---

# Evidence Hierarchy

Every statement in a dossier should belong to one of the following levels.

Level 1 — Observation

Direct measurements reported by an instrument, survey, or publication.

Level 2 — Derived Quantity

Physical parameters inferred from models (e.g., stellar mass, star-formation rate, dust temperature).

Level 3 — Published Interpretation

Scientific conclusions proposed by the original authors.

Level 4 — Einstein Synthesis

Connections between multiple publications produced by Einstein while preserving provenance.

Level 5 — Research Hypothesis

Ideas requiring future verification.

---

# Dossier Lifecycle

Stage 0
Object identified.

Stage 1
Primary literature summarized.

Stage 2
Cross-identifications completed.

Stage 3
Archive imagery collected.

Stage 4
Independent analyses performed.

Stage 5
AI-assisted synthesis complete.

Stage 6
Living scientific record.

---

## notes/

Working notes, hypotheses, calculations, and intermediate research.

These notes are exploratory and are not considered part of the formal dossier until incorporated into C19.md.

---

## papers/

Copies of papers directly relevant to the object.

Whenever possible, papers should retain their original filenames together with a brief description in the dossier.

---

# Scientific Standards

The dossier distinguishes carefully between:

## Observation

Measured quantities directly reported by surveys or publications.

## Derived Properties

Values calculated using astrophysical models.

## Interpretation

Scientific conclusions drawn by authors or by subsequent analysis.

## Hypotheses

Ideas requiring further verification.

These categories should never be mixed.

---

# Provenance

Every numerical value should be traceable.

Whenever practical, include:

- publication
- year
- table number
- figure number
- page number

This allows every statement in the dossier to be independently verified.

---

# Long-Term Vision

The Einstein Galaxy Dossier system is intended to become a digital encyclopedia of galaxies.

As additional dossiers are created, Einstein will assist with:

- literature discovery
- archive searches
- cross-identifications
- automatic updates
- comparative studies between galaxies
- generation of scientific reports and websites

The first prototype dossier is:

ASPECS-LP.1 mm C19

Future dossiers will follow the same structure, allowing the collection to grow into a consistent research resource.

---

*"Astronomy advances through careful observation, rigorous evidence, and continual refinement. These dossiers are designed to embody that philosophy while enabling effective collaboration between human researchers and AI."*