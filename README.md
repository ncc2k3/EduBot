# Chatbot_STSV
Hỗ trợ sinh viên IT tại HCMUS
--

### **Mục lục**
1. [Giới thiệu](#giới-thiệu)
2. [Tính năng](#tính-năng)
3. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
4. [Cài đặt](#cài-đặt)
5. [Cấu trúc dự án](#cấu-trúc-dự-án)

### **Giới thiệu**
Chatbot này được tạo ra nhằm...

### **Tính năng**
- Trả lời tự động các câu hỏi phổ biến.
- Kết nối với cơ sở dữ liệu để tìm kiếm thông tin.
- Hỗ trợ ngôn ngữ tự nhiên (NLP).

### **Yêu cầu hệ thống**

1. Ngôn ngữ lập trình
- Python: Phiên bản 3.10

2. Thư viện và công cụ chính
- **Langchain** | Framework hỗ trợ xây dựng ứng dụng tương tác với LLM.
- **Ollama** | Quản lý mô hình ngôn ngữ cục bộ (hỗ trợ Qwen2.5:7b).
- **Qwen 2.5:7B** | Mô hình ngôn ngữ lớn (LLM) mặc định được tích hợp.
- **Streamlit** | Xây dựng giao diện người dùng.

- **Chroma** | Vector database cho việc lưu trữ và truy xuất thông tin.
- **Huggingface Transformers** | Xử lý và tương tác với các mô hình ngôn ngữ lớn.

### **Cài đặt**
1. Clone repository
    ```
    git clone https://github.com/ncc2k3/Chatbot_STSV.git
    cd Chatbot_STSV
    ```
2. Requirements installization
    ```
    pip install -r requirements.txt
    ```
3. Run app
    ```
    streamlit run app.py
    ```

    **Cách sử dụng**
    - Sau khi chạy ứng dụng, truy cập giao diện tại http://localhost:8502.
    - Gõ câu hỏi hoặc yêu cầu vào ô nhập liệu để tương tác với chatbot.

### **Cấu trúc dự án**
```plaintext
Chatbot_STSV/
├── README.md           # Tài liệu hướng dẫn
├── app.py              # Tệp chính chạy ứng dụng Streamlit
├── requirements.txt    # Danh sách thư viện cần thiết
├── configs/            # Cấu hình ứng dụng
│   └── settings.py     # File cấu hình chính
├── data/               # Thư mục chứa dữ liệu
│   ├── docs/           # Các tài liệu sử dụng 
│   └── test/           # Bộ dữ liệu kiểm tra mô hình
├── src/                # Chứa các mã nguồn sử dụng
│   ├── base/           # Các mã nguồn dùng để khởi tạo mô hình
│   └── rag/            # Các mã nguồn cài đặt và vận hành RAG
│       └── chatbot.py  # Mã nguồn kiến trúc của chatbot
└── vectorstores/       # Chứa các vector embeddings từ văn bản
```
