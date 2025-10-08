import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# Khởi tạo state cho phiên chat nếu chưa có
if "chat_initialized" not in st.session_state:
    st.session_state["chat_initialized"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "initial_analysis_text" not in st.session_state:
    st.session_state["initial_analysis_text"] = ""
    
# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý giá trị 0 thủ công cho mẫu số để tính tỷ trọng
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm khởi tạo phiên Chat Gemini và tạo Nhận xét ban đầu ---
def initialize_gemini_chat_and_analysis(data_for_ai, api_key):
    """
    Khởi tạo phiên chat Gemini với System Instruction dựa trên dữ liệu,
    và gửi yêu cầu đầu tiên để lấy nhận xét ban đầu.
    """
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'

        # System Instruction: Thiết lập vai trò và cung cấp ngữ cảnh dữ liệu
        system_prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Hãy trả lời các câu hỏi của người dùng dựa trên dữ liệu tài chính mà bạn đã nhận được. Dữ liệu này bao gồm bảng cân đối kế toán đã được phân tích tăng trưởng và tỷ trọng, cùng với các chỉ số tài chính cơ bản.

        Dữ liệu phân tích chi tiết:
        {data_for_ai}

        Hãy duy trì ngữ cảnh này trong suốt cuộc trò chuyện.
        """
        
        # Khởi tạo phiên chat với System Instruction
        chat = client.chats.create(
            model=model_name,
            system_instruction=system_prompt,
        )
        st.session_state.chat_session = chat
        
        # Câu lệnh để Gemini tạo bản tóm tắt ban đầu (chức năng 5)
        initial_analysis_prompt = (
            "Dựa trên dữ liệu tài chính đã được cung cấp, hãy đưa ra một nhận xét khách quan, "
            "ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung "
            "vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành."
        )
        
        # Gửi tin nhắn đầu tiên để nhận phân tích
        response = chat.send_message(initial_analysis_prompt)
        initial_analysis_text = response.text
        
        # Lưu kết quả phân tích ban đầu vào lịch sử chat
        st.session_state.messages.append({"role": "assistant", "content": initial_analysis_text})
        
        # Đặt cờ đã khởi tạo thành True
        st.session_state.chat_initialized = True
        
        return initial_analysis_text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

# Reset session state khi tải file mới
if uploaded_file is not None and st.session_state.get('last_uploaded_file') != uploaded_file.name:
    st.session_state["chat_initialized"] = False
    st.session_state["messages"] = []
    st.session_state["initial_analysis_text"] = ""
    st.session_state['last_uploaded_file'] = uploaded_file.name
    
if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            # Khởi tạo giá trị mặc định cho Thanh toán Hiện hành
            thanh_toan_hien_hanh_N = "N/A"
            thanh_toan_hien_hanh_N_1 = "N/A"
            tsnh_growth_percent = "N/A"
            
            try:
                # Lấy Tài sản ngắn hạn
                tsnh_row = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]
                tsnh_n = tsnh_row['Năm sau'].iloc[0]
                tsnh_n_1 = tsnh_row['Năm trước'].iloc[0]
                tsnh_growth_percent = f"{tsnh_row['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%"

                # Lấy Nợ ngắn hạn
                no_ngan_han_row = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]
                no_ngan_han_N = no_ngan_han_row['Năm sau'].iloc[0] 
                no_ngan_han_N_1 = no_ngan_han_row['Năm trước'].iloc[0]

                # Tính toán
                if no_ngan_han_N != 0:
                     thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                if no_ngan_han_N_1 != 0:
                    thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, float) else thanh_toan_hien_hanh_N
                    )
                with col2:
                    delta_value = thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1 if isinstance(thanh_toan_hien_hanh_N, float) and isinstance(thanh_toan_hien_hanh_N_1, float) else None
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, float) else thanh_toan_hien_hanh_N,
                        delta=f"{delta_value:.2f}" if delta_value is not None else None
                    )
                    
            except IndexError:
                 st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
            except ZeroDivisionError:
                 st.warning("Không thể tính chỉ số Thanh toán Hiện hành do Nợ Ngắn Hạn bằng 0.")

            # --- Chức năng 5: Nhận xét AI (và Khởi tạo Chat) ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            if not st.session_state.chat_initialized:
                # Chuẩn bị dữ liệu để gửi cho AI (để làm System Instruction)
                data_for_ai = pd.DataFrame({
                    'Chỉ tiêu': [
                        'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                        'Tăng trưởng Tài sản ngắn hạn (%)', 
                        'Thanh toán hiện hành (N-1)', 
                        'Thanh toán hiện hành (N)'
                    ],
                    'Giá trị': [
                        df_processed.to_markdown(index=False),
                        tsnh_growth_percent, 
                        f"{thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N_1, float) else thanh_toan_hien_hanh_N_1, 
                        f"{thanh_toan_hien_hanh_N:.2f}" if isinstance(thanh_toan_hien_hanh_N, float) else thanh_toan_hien_hanh_N
                    ]
                }).to_markdown(index=False) 

                if st.button("Yêu cầu AI Phân tích & Bắt đầu Chat"):
                    api_key = st.secrets.get("GEMINI_API_KEY") 
                    
                    if api_key:
                        with st.spinner('Đang gửi dữ liệu, chờ Gemini phân tích và khởi tạo phiên chat...'):
                            ai_result = initialize_gemini_chat_and_analysis(data_for_ai, api_key)
                            st.session_state.initial_analysis_text = ai_result
                            st.rerun() # Chạy lại để hiển thị kết quả và khung chat
                    else:
                        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
            
            # Hiển thị kết quả phân tích ban đầu sau khi đã khởi tạo
            if st.session_state.chat_initialized:
                st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                st.info(st.session_state.initial_analysis_text)
                
            # --- Chức năng 6: Khung Chat Hỏi Đáp với AI ---
            if st.session_state.chat_initialized:
                st.divider()
                st.subheader("6. Hỏi đáp chuyên sâu với Gemini AI")
                st.caption("Hãy hỏi Gemini về tăng trưởng, cơ cấu, các chỉ số hoặc bất kỳ mục nào trong bảng dữ liệu.")
                
                # Hiển thị lịch sử chat
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Xử lý input từ người dùng
                if prompt := st.chat_input("Hỏi Gemini về báo cáo tài chính này..."):
                    
                    # 1. Thêm tin nhắn người dùng vào lịch sử và hiển thị
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # 2. Gửi tin nhắn đến phiên chat đang hoạt động
                    with st.chat_message("assistant"):
                        with st.spinner("Gemini đang phân tích câu hỏi..."):
                            try:
                                chat_session = st.session_state.chat_session
                                response = chat_session.send_message(prompt)
                                ai_response = response.text
                                
                                # 3. Thêm phản hồi của AI vào lịch sử và hiển thị
                                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                                st.markdown(ai_response)

                            except Exception as e:
                                error_message = f"Lỗi trong quá trình trò chuyện: {e}. Vui lòng thử lại."
                                st.error(error_message)
                                st.session_state.messages.append({"role": "assistant", "content": error_message})
                                st.markdown(error_message)


    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}. Vui lòng kiểm tra file Excel đảm bảo có chỉ tiêu 'TỔNG CỘNG TÀI SẢN' và 3 cột tiêu chuẩn.")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file và dữ liệu.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
