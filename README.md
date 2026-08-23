# Visa Conditions RAG Assistant

**Group ID:** 69

A Retrieval-Augmented Generation (RAG) system that gives international students accurate, source-grounded answers about their Australian student (subclass 500), Temporary Graduate (subclass 485), and Skills in Demand (subclass 482) visa conditions — reducing confusion and compliance risk.

Built as part of the WIL Project (COSC2669/COSC2816), following the Walert RAG pipeline and evaluation approach as a baseline.

## Team

| Student ID | Full Name | Role |
|---|---|---|
| S4182571 | Amogh Bandadi Mohan | Knowledge Base Lead |
| S4189448 | Chiranthan Channanja Swamy | RAG Pipeline Lead |
| S4164128 | Tanishq Deshpande | Evaluation & Testing Lead |
| S4175196 | Varnika Chandrashekar | Prototype & Frontend Lead |
| S4122887 | Yashas Raj Dinesh | Coordination & Docs Lead |

## Project Aim

We are building a Retrieval-Augmented Generation assistant that gives international students accurate, source-grounded answers about their Australian student (subclass 500), Temporary Graduate (subclass 485), and Skills in Demand (subclass 482) visa conditions, reducing confusion and compliance risk.

## Knowledge Base

- Department of Home Affairs pages for subclasses 500, 485, and 482 (eligibility, conditions, work rights, application steps)
- Study Melbourne guidance
- RMIT international student services guidance

## Approach

- Baseline pipeline reproduced from [Walert](https://github.com/rmit-ir/walert) (retrieval + generation over a curated corpus)
- Local LLM and embedding model served via [Ollama](https://ollama.com/), with a vector store for retrieval
- Evaluation framework covering effectiveness (% unanswered, accuracy), faithfulness (answer grounded in retrieved chunks), and source-attribution correctness
- Planned prototype: a minimal Streamlit interface

## Project Links

- **Trello board:** https://trello.com/invite/b/6a88095e471fbaf4184566ca/ATTI4e132bafd3f0a24ef1e5bc29bd867186DE884081/my-trello-board
- **Report:** see Canvas submission (WIL Project Milestone 1)

## Status

This repository is at Milestone 1: knowledge base scoping is complete and the baseline RAG pipeline is being set up. See the Milestone 1 report for full progress details and the plan for the next three weeks.
