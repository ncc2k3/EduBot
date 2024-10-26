import pdfplumber

def extract_and_format_text(pdf_path, output_txt_path):
    content = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Trích xuất văn bản từ trang
            text = page.extract_text()
            
            # Kiểm tra và xử lý bảng
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    table_content = "\n".join(
                        ["\t".join([str(cell) if cell else "" for cell in row]) for row in table]
                    )
                    content.append(f"--- Table from page {page_num + 1} ---\n")
                    content.append(table_content)
            
            # Xử lý nội dung văn bản
            if text:
                # Loại bỏ ngắt dòng sai và nối các đoạn văn
                clean_text = text.replace("-\n", "").replace("\n", " ")
                content.append(f"\n--- Page {page_num + 1} ---\n")
                content.append(clean_text)
    
    # Ghi kết quả ra file .txt
    with open(output_txt_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(content))

# Đường dẫn tới file PDF và file .txt để lưu
pdf_path = "data/STSV-2024-ONLINE.pdf"
output_txt_path = "data/output_cleaned.txt"

# Gọi hàm trích xuất và chuẩn hóa
extract_and_format_text(pdf_path, output_txt_path)
print(f"File đã được chuẩn hóa và lưu vào {output_txt_path}")
