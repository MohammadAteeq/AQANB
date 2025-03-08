# AQANB: Arabic Question Answering over Narrative Books

## Overview

AQANB is an Arabic Question Answering (QA) dataset specifically curated for extended narrative texts. This dataset enables the development and evaluation of QA systems capable of handling intricate narrative structures and broader contextual dependencies. Answering questions over long narratives is challenging due to the inherent token limitations of large language models, which restrict the amount of text processed in a single pass. Moreover, processing a larger number of tokens significantly increases computational costs. Additionally, processing only a selected set of relevant passages requires question-passage annotation to train the retriever. To address these challenges, we propose an annotation-free retriever training framework. Instead of relying on manually annotated data, our approach leverages large language models to generate synthetic question-passage pairs that probe deeper semantic content. Crucially, we introduce a novel narrative coherence-based scoring mechanism that assigns high relevance not only to the most pertinent passage but also to the surrounding passages, thereby capturing continuity and context rather than treating adjacent passages as purely negative samples. This design improves retrieval accuracy for narrative-style content where crucial information may span multiple, interconnected sections. Our experiments on AQANB demonstrate the effectiveness of the proposed approach. We achieve relative improvements of 21.7% and 11.7% over OpenAI’s small and large embedding models, respectively, and outperform the open-source multilingual Contriever by 29.6%. Furthermore, adding our retriever as a pre-stage to the common OpenAI GPT-4 reduced the cost by 98 times and achieved better performance compared to processing the entire book with GPT-4.

## Repository Contents

- **LongNarrativeBooks_Text/**: Contains text data from long narrative books.

- **QA_v103.3/**: Includes question-answer pairs for each book.

- **Books_V103.2.xlsx**: An Excel file that contains the details of books and specify which of used in training and testing.

- **passage_events_qa_10_100.zip**: A zipped file containing passage-level events and corresponding question-answer pairs.

- **train_bi_encoder.py**: A Python script for training a bi-encoder model

- **train_cross_encoder.py**: A Python script for training a cross-encoder model
