---
name: verify-thesis-paragraph
description: Verify, critique, and refine thesis or journal paragraphs against the full text of the cited research papers. Use whenever the user asks to revise, rewrite, strengthen, fact-check, source-check, cite, or review academic prose in this repository, including when the user supplies a paragraph without explicitly requesting citation verification. Also use to identify missing premises, stronger evidence, relevant papers, counter-evidence, or literature that the user needs to obtain.
---

# Verify Thesis Paragraph

## Overview

Apply this workflow to every academic paragraph under review. Treat source verification and argument completeness as part of the revision, not as optional follow-up work.

## Locate the working context

1. Treat `thesis/Reference/Research paper/` as the primary local paper collection.
2. Treat `thesis/DTR Thesis/Revised Thesis - Hybrid Depth-VLM Safety Guidance (HDSG) Framework A Safety-Constrained Architecture for Grounded Navigational Transparency in Smart Walkers.docx` as the single authoritative live thesis. Do not select another thesis file by date, filename similarity, or location.
3. Read the authoritative thesis directly for every review so that the quoted existing text comes from its latest saved state. If Word has unsaved changes, explain that only the last saved version is available and ask the user to save before relying on the document text.
4. Treat `thesis/ChatGPT Session/` as the only location for working drafts, verification notes, and other session artifacts produced by this skill.
5. Do not read, edit, copy from, or write files under `thesis/Claude Session/` unless the user explicitly requests access to that folder for the current task.
6. Establish the paragraph's role, terminology, research-question alignment, and surrounding argument from the live thesis and materials under `thesis/ChatGPT Session/`.
7. Never write to, modify, replace, or resave the authoritative thesis document. It contains live EndNote fields and is read-only for this workflow. Return proposed changes in chat or place working artifacts only under `thesis/ChatGPT Session/`.

## Verify every existing citation

For each citation or source-dependent claim:

1. Locate and open the actual paper. Use the PDF workflow available in the environment when the source is a PDF.
2. Search the full text for the central term, measurement, population, mechanism, or result in the claim.
3. Read enough surrounding material to establish context. Check the methods, results, table, figure, limitation, or discussion passage that carries the claim.
4. Confirm the author, year, study population, sample size, study design, variables, numerical values, comparison condition, and scope whenever they matter.
5. Distinguish the paper's own finding from a claim that its introduction attributes to another source.
6. Classify the citation as `supported`, `partially supported`, `unsupported`, or `unverified`.

Never verify a claim from a filename, title, abstract, search snippet, bibliography entry, previous summary, or plausibility. Never infer a negative result from one failed text search. Validate extraction against a term known to occur and account for line-break hyphenation.

If the full source is unavailable or unreadable, stop treating the claim as established. Mark it `[UNVERIFIED: reason]` and include the paper in the acquisition list. Do not invent a citation or silently substitute a different source.

## Audit the argument

Evaluate more than sentence quality. Determine whether the paragraph:

- states the job it performs in the section;
- contains the premises needed for its conclusion;
- explains the relevant mechanism rather than only naming it;
- distinguishes evidence, inference, and the thesis's own argument;
- represents comparisons, populations, and limitations fairly;
- acknowledges material counter-evidence or competing findings;
- avoids claiming novelty from absence without a defensible search basis;
- advances the chapter rather than repeating an earlier point;
- creates the required link to the next paragraph or section.

Check the surrounding thesis before adding material that may already be established elsewhere. Prefer a cross-reference over repetition.

## Strengthen the evidence

1. Search the local paper collection first for evidence that fills a genuine argumentative gap, qualifies an overbroad claim, provides a stronger comparison, or supplies a missing mechanism.
2. Read and verify any local paper before adding it to the revision.
3. If the required evidence is absent locally, search current scholarly sources on the web. Prefer the original research paper, systematic review, authoritative standard, or official publisher record appropriate to the claim.
4. Open and read the full text before presenting a new claim as verified.
5. If only metadata or an abstract is accessible, do not write the proposed claim as fact. Add the source to the acquisition list so the user can place the paper in `thesis/Reference/Research paper/`.

Add evidence only when it changes or completes the argument. Do not pad a paragraph with adjacent facts, decorative citations, or literature that belongs in another section.

## Write the revision

Apply any available project prose-style instructions alongside this skill. Otherwise use formal third-person thesis prose, British spelling, explicit logical transitions, and precise narrative or parenthetical citations.

Preserve the user's intended claim when the evidence supports it. Narrow, qualify, replace, or remove the claim when the evidence does not support it. Never retain a false or overstated sentence merely to minimise edits.

Do not attach a citation to a clause the source does not support. Use `[SOURCE NEEDED: description]` when the argument needs evidence but no verified source is available.

## Return the result

Provide these sections, omitting only sections that have no content:

1. **Verification verdict**: list each citation with its status and the relevant page, section, table, or figure where available.
2. **Argument findings**: identify missing premises, overstatement, contradiction, repetition, counter-evidence, or opportunities that materially improve the argument.
3. **Suggested replacement**: give a complete, ready-to-use paragraph, not disconnected line edits.
4. **Sources to obtain**: give the exact title, authors, year, venue, DOI or stable link, and the specific claim the source may support. Label the proposed use as unverified until the full paper has been read.

Keep the explanation concise. If reviewing text inside a Word document, identify the edit with a distinctive search string that can be pasted into Word's Find box, never a paragraph index.

For every proposed change to thesis text, use this sequence:

1. **Existing text to search**: quote the exact current sentence or complete paragraph from the live thesis, with enough text to locate it uniquely in Word.
2. **Scope of change**: state whether the change replaces one sentence, several named sentences, one complete paragraph, or a complete subsection. Describe the scope in structural terms, not word counts.
3. **Updated text**: provide the complete replacement sentence, paragraph, or subsection rather than only the altered fragment.

Never give a correction without the existing searchable text. If several changes affect one paragraph, quote the complete existing paragraph once and provide the complete revised paragraph once. If the user requests a full section, present the individual changes in this format and then provide the complete updated section.

## Non-negotiable evidence gate

Do not deliver a revised paragraph containing a source-dependent factual claim unless one of the following is true:

- the supporting passage in the full source has been read and verified; or
- the claim is visibly marked as needing a source or verification.

When verification is blocked, say exactly what file or paper is needed. Accuracy takes precedence over producing a seamless paragraph.
