import pdfplumber

# Bước 1: Đọc file PDF bằng pdfplumber
def load_pdf_with_plumber(filepath):
    documents = []
    with pdfplumber.open(filepath) as pdf:
        for page_number, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:  # Chỉ thêm trang nếu có nội dung
                documents.append(text)
    return documents

# Bước 2: Lưu nội dung PDF vào file .txt
def save_to_text_file(documents, output_filepath):
    with open(output_filepath, 'w', encoding='utf-8') as file:
        for page_content in documents:
            file.write(page_content)
            file.write('\n' + '-'*80 + '\n')  # Để phân cách giữa các trang

# Main function để chạy
def main():
    pdf_path = 'data/STSV-2024-ONLINE.pdf'  # Đường dẫn file PDF của bạn
    output_txt_path = 'data/output_pdf_content.txt'  # Đường dẫn lưu file .txt
    
    # Load PDF
    documents = load_pdf_with_plumber(pdf_path)
    
    # Lưu nội dung PDF vào file .txt
    save_to_text_file(documents, output_txt_path)
    print(f"Nội dung đã được lưu vào {output_txt_path}")

if __name__ == "__main__":
    main()
