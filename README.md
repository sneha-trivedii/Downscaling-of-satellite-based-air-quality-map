# 🛰️ Downscaling of Satellite-Based Air Quality Maps Using Machine Learning

> A complete end-to-end ML pipeline that transforms coarse-resolution TROPOMI satellite NO₂ data into high-resolution air quality maps over Delhi, India — achieving **R² = 0.9284** on unseen test data.

---

## 📌 Table of Contents

- [Live Demo](#live-demo)
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Research Question](#research-question)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Models Built](#models-built)
- [Results](#results)
- [Key Learnings](#key-learnings)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Future Work](#future-work)

---

## 🚀 Live Demo

Explore the interactive dashboard here: **[Live App →](https://downscaling-of-satellite-based-air-quality-map-fspkcqx4rjktp45.streamlit.app/)**

The dashboard includes 6 pages — Overview, Dataset & Pipeline, Models & Results, Final NO₂ Map, Key Learnings, and About & Links — with interactive Plotly charts and the final high-resolution Delhi NO₂ map.

---

## Project Overview

Satellites like **Sentinel-5P TROPOMI** monitor air pollution from space, but their data is coarse — each pixel covers approximately 3.5km × 3.5km. This makes it impossible to understand air quality at the neighborhood level.

This project uses **machine learning** to "downscale" this coarse satellite data — transforming blurry, low-resolution NO₂ maps into sharp, high-resolution maps that reveal street-level pollution patterns.

Think of it like using AI to sharpen a blurry photograph — except the photograph is a pollution map, and the sharpening is done using geographic and demographic context.
