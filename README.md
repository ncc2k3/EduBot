# FIT-HCMUS-EduBot

## Hỗ trợ sinh viên IT tại HCMUS

### **Mục lục**

1. [Giới thiệu](#giới-thiệu)
2. [Tính năng](#tính-năng)
3. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
4. [Cài đặt](#cài-đặt)
5. [Cấu trúc dự án](#cấu-trúc-dự-án)

### **Giới thiệu**

Edubot được tạo ra nhằm hỗ trợ sinh viên thuộc Khoa Công nghệ Thông tin trường Đại học Khoa học Tự nhiên (FIT-HCMUS) hỏi đáp về chương trình đào tạo và các thông tin cơ bản trong sổ tay sinh viên.

### **Tính năng**

- Hỏi đáp về chương trình đào tạo.
- Trả lời tự động các câu hỏi phổ biến.
- Kết nối với cơ sở dữ liệu để tìm kiếm thông tin.
- Hỗ trợ ngôn ngữ tự nhiên (NLP).

### **Yêu cầu hệ thống**

1. Ngôn ngữ lập trình

- Python: Phiên bản 3.10

2. Thư viện và công cụ chính

- **Langchain** | Framework hỗ trợ xây dựng ứng dụng tương tác với LLM.
- **Ollama** | Quản lý mô hình ngôn ngữ cục bộ (hỗ trợ Llama3.1:8B).
- **Streamlit** | Xây dựng giao diện người dùng.
- **Chroma** | Vector database cho việc lưu trữ và truy xuất thông tin.
- **Huggingface Transformers** | Xử lý và tương tác với các mô hình ngôn ngữ lớn.

### **Cài đặt**

1. Clone repository
   ```
   git clone https://github.com/ncc2k3/EduBot.git
   cd EduBot
   ```
2. Cài đặt các thư viện cần thiết
   ```
   pip install -r requirements.txt
   ```
   Đảm bảo rằng bạn đang sử dụng phiên bản pip mới nhất, để có thể nâng cấp lên phiên bản mới nhất, bản có thể sử dụng lệnh sau:
   ```
   python.exe -m pip install --upgrade pip
   ```
3. Thiết lập môi trường
   - Tạo file .env chứa nội dung có cấu trúc giống như file .env.examples
   - Thay đổi `GOOGLE API KEY` trong file .env thành api của bạn để sử dụng google api gọi mô hình bên ngoài
       
4. Khởi chạy ứng dụng
   ```
   streamlit run app.py
   ```

   **Cách sử dụng**

   - Sau khi chạy ứng dụng, truy cập giao diện tại http://localhost:8501.
   - Gõ câu hỏi hoặc yêu cầu vào ô nhập liệu để tương tác với chatbot.

   **Lưu ý**
   - Cần phải có google api key trong chứa trong file .env mới có thể sử dụng được mô hình Gemini
   - Bạn phải tải và cài đặt các mô hình trên Ollama trước để có thể sử dụng các mô hình khác ngoài Gemini 

### **Cấu trúc dự án**

```plaintext
EduBot/
├── README.md                 # Tài liệu hướng dẫn sử dụng chương trình
├── app.py                    # Tệp chính chạy ứng dụng Streamlit
├── requirements.txt          # Danh sách thư viện cần thiết
├── config.py                 # Cấu hình ứng dụng
├── prompt.py                 # Chứa các prompt hướng dẫn cho mô hình
├── data/                     # Thư mục chứa dữ liệu
│   ├── docs/                 # Các tài liệu sử dụng
│   ├── test/                 # Bộ dữ liệu kiểm tra mô hình
│   └── readme.md             # Mô tả về dataset
├── vectorstores/             # Chứa các vector embeddings từ văn bản
├── create_vectorstore.py     # Dùng để tạo vectorstore cho bộ dữ liệu
└── .env.examples             # Chứa cấu trúc ví dụ của file .env cần tạo để chạy chương trình

```
