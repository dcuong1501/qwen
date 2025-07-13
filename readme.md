
---


1. **Chọn phiên bản Python**:

   ```bash
   pyenv local 3.10.13
   ```

2. **Tạo môi trường ảo**:

   ```bash
   ~/.pyenv/versions/3.10.13/bin/python -m venv shakespeare_env
   ```

3. **Kích hoạt môi trường ảo**:

   ```bash
   source shakespeare_env/bin/activate
   ```

4. **Cài đặt phụ thuộc**:

   ```bash
   pip install -r requirements.txt
   ```


. **funetune**:

   ```bash
   funetune: python3 main.py
   ```

. **chạy chat**:

   ```bash
   python3 chat.py
   ```

