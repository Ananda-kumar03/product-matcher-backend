# product-matcher-backend
Python Flask Backend for Visual Product Matcher
This project implements a comprehensive full-stack fashion matching system. The Frontend is a React application hosted on Netlify, offering user authentication (Signup/Login) and product browsing. The Backend is a Python Flask API running on Render, utilizing MongoDB for user and product data management.

The core feature, Visual Search, is implemented using a machine learning pipeline:

Feature Extraction: The CLIP model (via Hugging Face Transformers) generates 512-dimension vector embeddings from a query image (URL or upload).

Indexing & Search: A pre-built FAISS index enables instantaneous k-Nearest Neighbors (kNN) search to find the most visually similar products among thousands.

The application works perfectly in the local development environment. However, deployment to the Render service failed due to memory constraints. The combined size of the PyTorch ML model, the Transformers library, and the large pre-built FAISS index file (~1GB) exceeds the RAM and disk limits of Render's free tier, preventing the model from loading into memory at startup.

Implemented Features
Feature	Technology	Status	Notes
Backend API	Flask, MongoDB, Render	Deployed (Base API)	Basic routes like Auth/Products load.
Frontend UI	React, Netlify	Deployed	User interface for browsing and search input.
User Management	MongoDB, bcrypt	Functional	Signup and Login routes with hashed passwords.
Product Browsing	MongoDB	Functional	Pagination for general product listing.
Visual Search	CLIP, FAISS, PyTorch	Functional Locally	Disabled on Render due to model memory limits.

Live Frontend URL	User Interface	Live	https://visualproductsmatcher.netlify.app/
Backend API URL	Render API (Base)	Live	https://visual-matcher-api-6qln.onrender.com
