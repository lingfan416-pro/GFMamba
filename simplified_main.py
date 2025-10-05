# main.py
import tkinter as tk
from tkinter import filedialog, messagebox
import shutil
import os
import time
import subprocess
import sys
from simplified_inference import main as inference_main


class ArchiverApp:
    def __init__(self, master):
        self.master = master
        master.title("Video File Analysis")

        # 文件路径变量  
        self.video_path = tk.StringVar()

        # 选择视频文件
        tk.Label(master, text="Select Video File:").grid(row=0, column=0)
        self.video_entry = tk.Entry(master, width=40, textvariable=self.video_path)
        self.video_entry.grid(row=0, column=1)
        browse_btn = tk.Button(master, text="Browse",
                               command=lambda: self.choose_file("video", self.video_entry))
        browse_btn.grid(row=0, column=2)

        # 归档按钮
        self.archive_button = tk.Button(master, text="Analyze File", command=self.archive_files)
        self.archive_button.grid(row=1, column=0, columnspan=3, pady=10)

        # 结果显示
        self.result_label = tk.Label(master, text="")
        self.result_label.grid(row=2, column=0, columnspan=3)

    def choose_file(self, filetype, entry):
        filetypes = [('Video files', '*.mp4;*.mov;*.avi')]
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            entry.delete(0, tk.END)
            entry.insert(0, filepath)

    def archive_files(self):
        video_path = self.video_path.get()
        if not video_path:
            messagebox.showerror("Error", "Please select a video file")
            return

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        temp_dir = os.path.abspath(f'temp_{timestamp}')
        downloads_dir = os.path.expanduser("~/Downloads/archived_videos")
        final_dir = os.path.join(downloads_dir, f'archive_{timestamp}')

        try:
            # 创建临时目录
            os.makedirs(temp_dir, exist_ok=True)
            shutil.copy(video_path, temp_dir)

            # 运行情感分析
            video_name = os.path.basename(video_path)
            archived_video_path = os.path.join(temp_dir, video_name)
            inference_main(archived_video_path)

            # 创建下载目录并移动文件
            os.makedirs(downloads_dir, exist_ok=True)
            shutil.copytree(temp_dir, final_dir)

            # 删除临时文件夹
            shutil.rmtree(temp_dir)

            self.result_label.config(text=f"Analysis Complete: {final_dir}")

            # 打开文件夹按钮指向新路径
            open_btn = tk.Button(self.master, text="Open Folder", command=lambda: self.open_folder(final_dir))
            open_btn.grid(row=3, column=0, columnspan=3)

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")

    def open_folder(self, folder):
        if sys.platform == "win32":
            os.startfile(folder)
        else:
            subprocess.Popen(["open", folder])


if __name__ == "__main__":
    root = tk.Tk()
    app = ArchiverApp(root)
    root.mainloop()
