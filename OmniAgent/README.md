# OmniAgent: Context-Aware Customer Service

## Project Overview

OmniAgent is an open-source framework that leverages PyTorch and LangChain to build intelligent chatbots for customer service automation. Using a Retrieval-Augmented Generation (RAG) approach, OmniAgent creates responsive, context-aware, and personalized customer interactions.

Built with The University of Texas at Dallas as our knowledge base, OmniAgent demonstrates how organizations can transform traditional customer service systems into dynamic conversational interfaces. The system uses Google's Flan T5 models with HuggingFace Instructor Embeddings and FAISS vector storage to provide accurate, contextual responses to user queries.

## Features

- Web scraping for automated data collection
- PyTorch implementation for model operations
- Vector embeddings for semantic understanding
- FAISS integration for similarity search
- Conversation memory for multi-turn interactions
- Gradio-based user interface

## Setup

1.  Clone the repository:

    bash

    ```
    git clone https://github.com/jaysuthar/omniagent.git
    cd omniagent
    ```

2.  Install dependencies:

    bash

    ```
    pip install -r requirements.txt
    ```

3.  Configure the website URL in `config/config.yaml`:

    yaml

    ```
    scraping:
      base_url: "https://www.utdallas.edu"
      max_pages: 100
    ```

4.  Run the web scraper:

    bash

    ```
    python src/data_collection/web_scraper.py
    ```

5.  Generate embeddings:

    bash

    ```
    python src/embeddings/instructor_embeddings.py
    ```

6.  Start the Gradio interface:

    bash

    ```
    python src/interface/gradio_app.py
    ```

## Performance Matrix

| Model         | Query Understanding | Response Accuracy | Response Time | Context Retention |
| ------------- | ------------------- | ----------------- | ------------- | ----------------- |
| Flan T5 XXL   | ★★★★★               | ★★★★★             | ★★★           | ★★★★★             |
| Flan T5 Base  | ★★★★                | ★★★               | ★★★★          | ★★★               |
| Flan T5 Small | ★★★                 | ★★                | ★★★★★         | ★★                |

## Author

Jay Suthar (jaysutharswe@gmail.com)

## License

MIT License

Copyright (c) 2025
