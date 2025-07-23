# 📄 Document Similarity Analyzer

![Project Banner](https://github.com/your-username/document-similarity/blob/main/assets/banner.gif?raw=true)

<div align="center">
  
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

## 🌟 Features

<div align="center">
  
| Feature | Description | Preview |
|---------|-------------|---------|
| **Multi-Format Support** | Process TXT, PDF, DOCX & MD files | ![File Upload](https://github.com/your-username/document-similarity/blob/main/assets/upload-demo.gif?raw=true) |
| **Smart Analysis** | NLP-powered similarity detection | ![Analysis](https://github.com/your-username/document-similarity/blob/main/assets/analysis-demo.gif?raw=true) |
| **Interactive Visuals** | Explore data with beautiful charts | ![Visualization](https://github.com/your-username/document-similarity/blob/main/assets/viz-demo.gif?raw=true) |
| **Query Search** | Find similar content with natural language | ![Search](https://github.com/your-username/document-similarity/blob/main/assets/search-demo.gif?raw=true) |

</div>

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/DocumentSimilarity.git
cd DocumentSimilarity

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
cd frontend
streamlit run app.py
```

The application will automatically open in your browser at `http://localhost:8501`

## 🛠️ Project Structure

```mermaid
graph TD
    A[Document Similarity] --> B[Frontend]
    A --> C[Backend]
    B --> D[Streamlit Interface]
    C --> E[Document Processor]
    C --> F[Similarity Engine]
    C --> G[Data Management]
    E --> H[Text Extraction]
    E --> I[Embedding Generation]
    F --> J[Pairwise Comparison]
    F --> K[Threshold Filtering]
    G --> L[Input Documents]
    G --> M[Processed Cache]
    G --> N[Results Storage]
```

## 📊 Technology Stack

<div align="center">
  
| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Backend | Python 3.8+ |
| NLP Model | all-MiniLM-L6-v2 |
| Text Processing | PyPDF2, python-docx |
| Visualization | Plotly, Matplotlib |
| Data Management | Pandas, Joblib |

</div>

## 📈 Performance Metrics

```mermaid
pie
    title Processing Time Distribution
    "Text Extraction" : 25
    "Embedding Generation" : 45
    "Similarity Calculation" : 20
    "Result Compilation" : 10
```

## 📚 Usage Guide

### 1. Document Upload
- Supports batch upload of multiple files
- Automatic format detection
- File size validation (max 16MB)

### 2. Analysis Configuration
```mermaid
graph LR
    A[Set Threshold] --> B(0.1-0.3: Loose)
    A --> C(0.4-0.6: Moderate)
    A --> D(0.7-1.0: Strict)
```

### 3. Results Interpretation
- Similarity scores range from 0 (no similarity) to 1 (identical)
- Color-coded results for easy scanning
- Export options for further analysis

## 🤝 Contributing

We welcome contributions! Please follow these steps:
1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
