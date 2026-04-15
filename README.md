rag-demo/
├── .venv/
├── requirements.in
├── requirements.txt
├── src/
│   ├── main.py              # FastAPI入口
│   ├── rag/
│   │   ├── pipeline.py      # RAG逻辑
│   │   ├── vector_store.py  # 向量库
│   │   └── loader.py        # 文档加载
│   └── data/
│       └── docs.txt         # 知识库
├── web/
│   └── index.html           # 前端
└── README.md