
---

# Hướng dẫn cài đặt môi trường ảo

1. **Cài đặt `pyenv`** (nếu chưa cài):

   * Làm theo hướng dẫn tại [pyenv GitHub](https://github.com/pyenv/pyenv).

2. **Chọn phiên bản Python**:

   ```bash
   pyenv local 3.10.13
   ```

3. **Tạo môi trường ảo**:

   ```bash
   ~/.pyenv/versions/3.10.13/bin/python -m venv shakespeare_env
   ```

4. **Kích hoạt môi trường ảo**:

   ```bash
   source shakespeare_env/bin/activate
   ```

5. **Cài đặt phụ thuộc**:

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

