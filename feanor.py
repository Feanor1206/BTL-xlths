import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import soundfile as sf
import os
import math
class BiquadFilter:
    def __init__(self):
        self.b0, self.b1, self.b2 = 1.0, 0.0, 0.0
        self.a1, self.a2 = 0.0, 0.0
        self.w_n_1 = 0.0  # w[n-1]
        self.w_n_2 = 0.0  # w[n-2]

    def set_coeffs(self, b0, b1, b2, a1, a2):
        self.b0, self.b1, self.b2 = b0, b1, b2
        self.a1, self.a2 = a1, a2

    def process(self, x_n):
        w_n = x_n - self.a1 * self.w_n_1 - self.a2 * self.w_n_2
        y_n = self.b0 * w_n + self.b1 * self.w_n_1 + self.b2 * self.w_n_2
        self.w_n_2 = self.w_n_1
        self.w_n_1 = w_n
        return y_n

class StereoEQChain_Offline:
    def __init__(self, fs):
        self.fs = fs
        self.bands = [
            {'type': 'lowshelf',  'f0': 100.0,  'Q': 0.7, 'G': 0.0}, # Bass
            {'type': 'peaking',   'f0': 400.0,  'Q': 1.0, 'G': 0.0}, # Low-Mid
            {'type': 'peaking',   'f0': 1500.0, 'Q': 1.5, 'G': 0.0}, # Mid
            {'type': 'peaking',   'f0': 5000.0, 'Q': 2.0, 'G': 0.0}, # High-Mid
            {'type': 'highshelf', 'f0': 10000.0, 'Q': 0.7, 'G': 0.0}, # Treble
        ]
        
        self.filters_L = [BiquadFilter() for _ in self.bands]
        self.filters_R = [BiquadFilter() for _ in self.bands]

    def set_gains(self, gain_list_db):
        for i, gain_db in enumerate(gain_list_db):
            self.bands[i]['G'] = gain_db
        self.recalculate_all_coeffs()

    def calculate_coeffs(self, band):
        G, f0, Q, filter_type = band['G'], band['f0'], band['Q'], band['type']
        
        A = 10**(G / 40.0) if filter_type == 'peaking' else 10**(G / 20.0)
        w0 = 2 * math.pi * f0 / self.fs
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        
        if Q <= 0: Q = 1e-16 # Đảm bảo Q > 0
        if self.fs <= 0: return 1.0, 0.0, 0.0, 0.0, 0.0 # An toàn

        if filter_type == 'peaking':
            alpha = sin_w0 / (2.0 * Q)
            b0 = 1 + alpha * A; b1 = -2 * cos_w0; b2 = 1 - alpha * A
            a0 = 1 + alpha / A; a1 = -2 * cos_w0; a2 = 1 - alpha / A
        
        elif filter_type == 'lowshelf':
            alpha = sin_w0 / (2 * Q) * (A**0.5)
            b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * (A**0.5) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
            b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * (A**0.5) * alpha)
            a0 = (A + 1) + (A - 1) * cos_w0 + 2 * (A**0.5) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
            a2 = (A + 1) + (A - 1) * cos_w0 - 2 * (A**0.5) * alpha
            
        elif filter_type == 'highshelf':
            alpha = sin_w0 / (2 * Q) * (A**0.5)
            b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * (A**0.5) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
            b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * (A**0.5) * alpha)
            a0 = (A + 1) - (A - 1) * cos_w0 + 2 * (A**0.5) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
            a2 = (A + 1) - (A - 1) * cos_w0 - 2 * (A**0.5) * alpha
        else: # Pass-through
            b0, b1, b2, a0, a1, a2 = 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
        
        if a0 == 0.0: a0 = 1e-16 # Tránh chia cho 0
        
        return b0/a0, b1/a0, b2/a0, a1/a0, a2/a0

    def recalculate_all_coeffs(self):
        for i, band in enumerate(self.bands):
            b0, b1, b2, a1, a2 = self.calculate_coeffs(band)
            self.filters_L[i].set_coeffs(b0, b1, b2, a1, a2)
            self.filters_R[i].set_coeffs(b0, b1, b2, a1, a2)

    def process_signal_offline(self, signal_in):
        if signal_in.ndim == 1:
            signal_in = np.stack([signal_in, signal_in], axis=1)
        
        signal_out = np.zeros_like(signal_in)
        num_samples = len(signal_in)
      
        for i in range(num_samples):
            # Lấy mẫu L/R
            sample_L = signal_in[i, 0]
            sample_R = signal_in[i, 1]
            
            for filt_L in self.filters_L:
                sample_L = filt_L.process(sample_L)
                
            for filt_R in self.filters_R:
                sample_R = filt_R.process(sample_R)
            
            signal_out[i, 0] = sample_L
            signal_out[i, 1] = sample_R
            
        return signal_out


class OfflineEQApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BTL Xử lý tín hiệu số - Made by Feanor")
        self.root.geometry("600x450")

        self.audio_data = None
        self.fs = 0
        self.input_filepath = ""

        # Frame chính
        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(expand=True, fill=tk.BOTH)

        # 1. Điều khiển File
        file_frame = ttk.LabelFrame(main_frame, text="Điều khiển")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_load = ttk.Button(file_frame, text="1. Import File Nhạc", command=self.load_file)
        self.btn_load.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

        self.btn_process = ttk.Button(file_frame, text="2. Xử lý & Export File", command=self.process_file, state=tk.DISABLED)
        self.btn_process.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        self.status_label = ttk.Label(main_frame, text="Chưa tải file...")
        self.status_label.pack(fill=tk.X, padx=10, pady=5)

        # 2. Bảng điều khiển EQ 5-Band
        eq_frame = ttk.LabelFrame(main_frame, text="Bảng điều khiển EQ 5-Band")
        eq_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        self.sliders = []
        self.slider_labels = []
        
        # Các dải tần (tên, f0)
        bands_info = [
            ("Low Shelf (100Hz)", 0),
            ("Low-Mid (400Hz)", 1),
            ("Mid (1500Hz)", 2),
            ("High-Mid (5000Hz)", 3),
            ("High Shelf (10kHz)", 4)
        ]
        
        for name, index in bands_info:
            slider_frame = ttk.Frame(eq_frame)
            slider_frame.pack(fill=tk.X, padx=10, pady=10)
            
            label = ttk.Label(slider_frame, text=f"{name:18s}", width=20)
            label.pack(side=tk.LEFT)
            
            # Thanh trượt từ -12dB đến +12dB
            slider = ttk.Scale(slider_frame, from_=-12.0, to=12.0, orient=tk.HORIZONTAL,
                               command=lambda v, i=index: self.on_slider_change(i, v))
            slider.set(0.0) # Giá trị mặc định
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            value_label = ttk.Label(slider_frame, text=" 0.0 dB", width=8)
            value_label.pack(side=tk.LEFT)
            
            self.sliders.append(slider)
            self.slider_labels.append(value_label)

    def on_slider_change(self, index, value):
        gain_db = float(value)
        self.slider_labels[index].config(text=f"{gain_db:6.1f} dB")

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.flac *.ogg"), ("All Files", "*.*")])
        if not file_path:
            return
            
        try:
            self.audio_data, self.fs = sf.read(file_path, dtype='float32')
            self.input_filepath = file_path
            
            self.status_label.config(text=f"Đã tải: {os.path.basename(file_path)} | {self.fs} Hz | {self.audio_data.shape}")
            self.btn_process.config(state=tk.NORMAL)
                
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể đọc file: {e}")
            self.btn_process.config(state=tk.DISABLED)

    def process_file(self):
        if self.audio_data is None:
            messagebox.showerror("Lỗi", "Chưa tải file nhạc!")
            return

        # Hỏi người dùng nơi lưu file
        save_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV File", "*.wav"), ("FLAC File", "*.flac")],
            initialfile=f"{os.path.splitext(os.path.basename(self.input_filepath))[0]}_eq"
        )
        if not save_path:
            return

        gains_db = [slider.get() for slider in self.sliders]
        
        self.status_label.config(text="Đang xử lý... Vui lòng chờ...")
        self.root.update_idletasks() # Cập nhật GUI ngay lập tức

        try:
            eq = StereoEQChain_Offline(fs=self.fs)
            eq.set_gains(gains_db)
            processed_data = eq.process_signal_offline(self.audio_data)
            max_val = np.max(np.abs(processed_data))
            if max_val > 1.0:
                processed_data = processed_data / max_val
                print(f"Cảnh báo: Tín hiệu bị clipping! Đã tự động chuẩn hóa (Peak: {max_val:.2f})")
  
            sf.write(save_path, processed_data, self.fs)
            
            self.status_label.config(text=f"Xong! Đã lưu tại: {save_path}")
            messagebox.showinfo("Thành công", "Đã xử lý và lưu file thành công!")

        except Exception as e:
            messagebox.showerror("Lỗi xử lý", f"Đã xảy ra lỗi: {e}")
            self.status_label.config(text="Xử lý thất bại!")

# Tinh hoa hội tụ ở đây, cuối cùng đã xong 
if __name__ == "__main__":
    try:
        import numpy
        import soundfile
    except ImportError:
        print("LỖI: Vui lòng cài đặt các thư viện cần thiết.")
        print("Chạy lệnh: pip install numpy soundfile")
        exit()

    root = tk.Tk()
    app = OfflineEQApp(root)
    root.mainloop()